import os
import glob
import random
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import gc

import numpy as np
import torch
import json

def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ----------------------------
# Global caches 
# ----------------------------
_FILELIST_CACHE: Dict[str, List[str]] = {}
# LRU to avoid blowing RAM if dataset huge
_AUDIO_CACHE: "OrderedDict[str, Tuple[torch.Tensor,int]]" = OrderedDict()
_AUDIO_CACHE_MAX_FILES = 512  # adjust if needed

_VIEW_CACHE: Dict[Tuple[str, str], "ViewModel"] = {}          # (device, view_name)
_ATTACKER_CACHE: Dict[Tuple[str, str], "AttackRouter"] = {}   # (device, attack_name)


def list_audio_files(audio_dir: str) -> List[str]:
    # cache glob results (I/O + recursion)
    if audio_dir in _FILELIST_CACHE:
        return _FILELIST_CACHE[audio_dir]

    files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True) + \
            glob.glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True)
    files = sorted(files)
    _FILELIST_CACHE[audio_dir] = files
    return files



def cached_load_audio(fp: str) -> Tuple[torch.Tensor, int]:
    """
    Cache torchaudio.load results (decode only once per file).
    LRU eviction to cap memory.
    """
    if fp in _AUDIO_CACHE:
        wav, sr = _AUDIO_CACHE.pop(fp)
        _AUDIO_CACHE[fp] = (wav, sr)  # move to end (most recent)
        return wav, sr

    wav, sr = torchaudio.load(fp)
    wav = ensure_mono(wav)
    wav = crop_loudest_segment(wav, sr, seconds=5.0)  # <- 只 cache 5 秒

    # keep CPU tensor; most ops later move/resample anyway
    _AUDIO_CACHE[fp] = (wav, sr)
    if len(_AUDIO_CACHE) > _AUDIO_CACHE_MAX_FILES:
        _AUDIO_CACHE.popitem(last=False)  # evict LRU
    return wav, sr


def clear_dataset_audio_cache():
    _AUDIO_CACHE.clear()
    # 檔名列表 cache 留著也行；但你要更乾淨也可：
    # _FILELIST_CACHE.clear()


def get_attacker(device: str, attack_name: str) -> "AttackRouter":
    key = (device, attack_name.lower())
    if key not in _ATTACKER_CACHE:
        _ATTACKER_CACHE[key] = AttackRouter(device, attack_name)
    return _ATTACKER_CACHE[key]

def _is_oom(e: Exception) -> bool:
    s = str(e).lower()
    return ("out of memory" in s) or ("cuda out of memory" in s)

def clear_model_caches():
    _ATTACKER_CACHE.clear()
    _VIEW_CACHE.clear()
    cuda_cleanup()

# ----------------------------
# Helpers
# ----------------------------
def ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    # Accept (C,T) or (B,C,T)
    if wav.dim() == 2 and wav.size(0) > 1:
        return wav.mean(dim=0, keepdim=True)
    if wav.dim() == 3 and wav.size(1) > 1:
        return wav.mean(dim=1, keepdim=True)
    return wav

def ensure_channels_bct(w_bct: torch.Tensor, n_ch: int) -> torch.Tensor:
    """
    w_bct: (B,C,T) -> force C == n_ch
    - if C==1 and n_ch==2: duplicate channel
    - if C==2 and n_ch==1: average to mono
    - else: truncate or repeat first channel to fill
    """
    assert w_bct.dim() == 3, f"expected (B,C,T), got {tuple(w_bct.shape)}"
    C = int(w_bct.shape[1])
    if C == n_ch:
        return w_bct
    if C == 1 and n_ch == 2:
        return w_bct.repeat(1, 2, 1)
    if C == 2 and n_ch == 1:
        return w_bct.mean(dim=1, keepdim=True)
    if C > n_ch:
        return w_bct[:, :n_ch, :]
    # C < n_ch: pad by repeating first channel
    rep = n_ch - C
    extra = w_bct[:, :1, :].repeat(1, rep, 1)
    return torch.cat([w_bct, extra], dim=1)


def as_bct(wav: torch.Tensor) -> torch.Tensor:
    """
    Force waveform to shape (B, C, T).
    Accept (T), (C,T), (B,T), (B,C,T), (B,C,1,T)
    """
    if wav.dim() == 1:
        wav = wav[None, None, :]
    elif wav.dim() == 2:
        # (C,T) or (B,T). if first dim small, treat as channels
        if wav.shape[0] <= 2:
            wav = wav[None, :, :]
        else:
            wav = wav[:, None, :]
    elif wav.dim() == 3:
        pass
    elif wav.dim() == 4 and wav.shape[2] == 1:
        wav = wav.squeeze(2)
    else:
        raise RuntimeError(f"Unexpected wav shape: {tuple(wav.shape)}")
    return wav

def match_length(out_bct: torch.Tensor, T0: int) -> torch.Tensor:
    if out_bct.shape[-1] > T0:
        return out_bct[..., :T0]
    if out_bct.shape[-1] < T0:
        return F.pad(out_bct, (0, T0 - out_bct.shape[-1]))
    return out_bct

def pad_to_multiple(w_bct: torch.Tensor, m: int = 4096) -> torch.Tensor:
    r = w_bct.shape[-1] % m
    if r != 0:
        w_bct = F.pad(w_bct, (0, m - r))
    return w_bct

def pad_for_encodec_chunking(w_bct: torch.Tensor, cfg) -> torch.Tensor:
    """
    HF EnCodec chunked mode needs:
      input_length % stride == (chunk_length - stride)
    where stride=cfg.chunk_stride, chunk_length=cfg.chunk_length.
    """
    chunk_length = getattr(cfg, "chunk_length", None)
    chunk_stride = getattr(cfg, "chunk_stride", None)
    if chunk_length is None or chunk_stride is None:
        return w_bct

    chunk_length = int(chunk_length)
    chunk_stride = int(chunk_stride)
    if chunk_stride <= 0:
        return w_bct

    T = int(w_bct.shape[-1])
    T1 = max(T, chunk_length)
    step = (chunk_length - chunk_stride) % chunk_stride
    rem = T1 % chunk_stride
    pad2 = (step - rem) % chunk_stride
    pad_total = (T1 + pad2) - T
    if pad_total > 0:
        w_bct = F.pad(w_bct, (0, pad_total))
    return w_bct

def safe_snac_roundtrip(model, w_bct: torch.Tensor, max_tries: int = 12, pad_step: int = 512) -> torch.Tensor:
    """
    SNAC encode/decode sometimes crashes due to window alignment.
    We pad-and-retry until it works.
    """
    w_try = w_bct
    step = pad_step
    last_err = None
    for i in range(max_tries):
        try:
            codes = model.encode(w_try)
            rec = model.decode(codes)
            if rec.dim() == 2:
                rec = rec.unsqueeze(0)  # (1,C,T)
            return rec
        except (EinopsError, RuntimeError) as e:
            last_err = e
            w_try = F.pad(w_try, (0, step))
            if i in (3, 7):
                step *= 2
    raise RuntimeError(f"SNAC roundtrip keeps failing after retries. Last error: {last_err}")

def calibrate_threshold_fpr(
    wm,
    audio_dir: str,
    calib_files: int,
    seed: int = 0,
    fpr: float = 0.01,
    seconds: float = 5.0,
) -> Dict[str, float]:
    """
    Calibrate threshold on clean audio:
      NEG: score = detect(original)
      POS: score = detect(watermarked)
    Decide direction automatically:
      if mean(POS) > mean(NEG): "higher => watermarked"
      else: "lower => watermarked"

    Control FPR on NEG:
      direction=+1: thr = quantile(NEG, 1-fpr), pass if score > thr
      direction=-1: thr = quantile(NEG, fpr),   pass if score < thr
    """
    files = list_audio_files(audio_dir)

    
    if not files:
        raise RuntimeError(f"No audio files for calibration: {audio_dir}")

    rnd = random.Random(seed)
    rnd.shuffle(files)
    files = files[:min(calib_files, len(files))]

    neg_scores, pos_scores = [], []
    used = 0

    for fp in tqdm(files, desc=f"CalibThresh {wm.name}", leave=False):
        try:
            wav, sr = cached_load_audio(fp)

            T = int(sr * seconds)
            if wav.shape[-1] < T:
                rep = (T // max(1, wav.shape[-1])) + 1
                wav = wav.repeat(1, rep)
            wav = wav[..., :T]

            # NEG (clean)
            s_neg = float(wm.detect(wav, sr, payload=None))
            neg_scores.append(s_neg)

            # POS (watermarked)
            wm_wav, payload = wm.embed(wav, sr)
            s_pos = float(wm.detect(wm_wav, wm.wm_sr, payload))
            pos_scores.append(s_pos)

            used += 1
        except Exception:
            continue

    if used < 5 or len(neg_scores) < 5:
        raise RuntimeError(f"Too few valid files for threshold calibration: used={used}")

    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = np.asarray(pos_scores, dtype=np.float64)

    pos_mean = float(pos.mean())
    neg_mean = float(neg.mean())

    direction = 1.0 if pos_mean >= neg_mean else -1.0

    if direction > 0:
        thr = float(np.quantile(neg, 1.0 - fpr))
        fpr_emp = float((neg > thr).mean())
        tpr_emp = float((pos > thr).mean())
    else:
        thr = float(np.quantile(neg, fpr))
        fpr_emp = float((neg < thr).mean())
        tpr_emp = float((pos < thr).mean())

    return {
        "threshold": thr,
        "direction": float(direction),
        "fpr_target": float(fpr),
        "fpr_emp": fpr_emp,
        "tpr_emp": tpr_emp,
        "neg_mean": float(neg.mean()),
        "neg_std": float(neg.std(ddof=0)),
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std(ddof=0)),
        "N": float(len(neg)),
    }


def crop_loudest_segment(wav_ct: torch.Tensor, sr: int, seconds: float = 5.0, hop_s: float = 0.25) -> torch.Tensor:
    """
    wav_ct: (C,T) mono preferred; returns (C, win) with win=sr*seconds
    deterministic: pick the window with max energy (mean square) evaluated every hop_s seconds.
    """
    assert wav_ct.dim() == 2, f"expected (C,T), got {tuple(wav_ct.shape)}"
    T = int(wav_ct.shape[-1])
    win = int(sr * seconds)
    if win <= 0:
        return wav_ct

    # If too short: repeat to at least win
    if T < win:
        rep = (win // max(1, T)) + 1
        wav_ct = wav_ct.repeat(1, rep)[:, :win]
        return wav_ct

    x = wav_ct[0]  # mono channel
    e = x * x
    # prefix sum with cs0[0]=0, cs0[t]=sum_{0..t-1}
    cs0 = torch.zeros(T + 1, dtype=e.dtype, device=e.device)
    cs0[1:] = torch.cumsum(e, dim=0)

    hop = max(1, int(sr * hop_s))
    idxs = torch.arange(0, T - win + 1, hop, device=e.device)
    # energy of each window: sum(e[start:start+win]) = cs0[start+win] - cs0[start]
    energies = cs0[idxs + win] - cs0[idxs]
    best = int(idxs[int(torch.argmax(energies))].item())
    return wav_ct[:, best: best + win]



# ----------------------------
# Codec loaders (SNAC / EnCodec / DAC)
# ----------------------------
try:
    from snac import SNAC
except Exception as e:
    SNAC = None
    print("[WARN] snac not available:", e)

try:
    from audiocraft.models import CompressionModel
except Exception as e:
    CompressionModel = None
    print("[WARN] audiocraft not available:", e)

try:
    import dac
except Exception as e:
    dac = None
    print("[WARN] descript-audio-codec (dac) not available:", e)

# einops error class for SNAC window-attn crash
try:
    from einops import EinopsError
except Exception:
    EinopsError = Exception

def safe_snac_encoder_out(model, w_bct: torch.Tensor, max_tries: int = 12, pad_step: int = 512) -> torch.Tensor:
    """
    SNAC encoder sometimes requires internal token length divisible by a fixed 'windows'.
    We don't guess alignment; we pad waveform and retry until encoder doesn't crash.
    Return z in (B, D, T) format.
    """
    w_try = w_bct
    step = pad_step
    last_err = None
    for i in range(max_tries):
        try:
            z = model.encoder(w_try)[0]
            if z.dim() == 2:
                z = z.unsqueeze(0)

            # SNAC encoder output might be (B,T,D) -> convert to (B,D,T)
            # Typical failing case inside attention: (B, 298, 1024) => likely (B,T,D)
            if z.dim() == 3 and z.shape[1] < z.shape[2]:
                z = z.transpose(1, 2).contiguous()
            return z
        except (EinopsError, RuntimeError) as e:
            last_err = e
            w_try = F.pad(w_try, (0, step))
            if i in (3, 7):
                step *= 2
    raise RuntimeError(f"SNAC encoder keeps failing after retries. Last error: {last_err}")


@dataclass
class ViewModel:
    name: str
    sr: int
    def latent(self, w_bct: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def codec_roundtrip(self, w_bct: torch.Tensor) -> torch.Tensor: raise NotImplementedError

class SNACView(ViewModel):
    def __init__(self, hub: str, sr: int, device: str, name: str):
        if SNAC is None: raise RuntimeError("snac not installed.")
        self.name = name; self.sr = sr; self.device = device
        self.model = SNAC.from_pretrained(hub).to(device).eval()
        for p in self.model.parameters(): p.requires_grad = False

    def _safe_encoder(self, w_bct):
        max_tries = 6
        pad_step = 256 # 每次多補 256 點
        
        last_err = None
        for i in range(max_tries):
            try:
                # 嘗試加上額外的 Padding
                if i > 0:
                    pad_amt = pad_step * i
                    w_curr = F.pad(w_bct, (0, pad_amt))
                else:
                    w_curr = w_bct
                
                z = self.model.encoder(w_curr)
                if isinstance(z, (list, tuple)): z = z[0]
                return z
            
            except Exception as e:
                # check if error is likely due to shape mismatch (EinopsError or related)
                err_str = str(e)
                if "EinopsError" in str(type(e)) or "shape" in err_str.lower() or "mismatch" in err_str.lower():
                    last_err = e
                    continue # Try next padding
                raise e # Other errors should not be retried
        
        raise RuntimeError(f"SNAC latent extraction failed after {max_tries} padding attempts. Last error: {last_err}")


    def latent(self, w_bct: torch.Tensor) -> torch.Tensor:
        # w_bct: (B,C,T)
        w_bct = ensure_mono(as_bct(w_bct)).to(self.device)
        w_bct = pad_to_multiple(w_bct, 4096)
        z = safe_snac_encoder_out(self.model, w_bct, max_tries=12, pad_step=512)  # (B,D,T)
        return z

    @torch.no_grad()
    def codec_roundtrip(self, w_bct: torch.Tensor) -> torch.Tensor:
        w_bct = ensure_mono(as_bct(w_bct)).to(self.device)
        w_bct = pad_to_multiple(w_bct, 4096)
        rec = safe_snac_roundtrip(self.model, w_bct, max_tries=12, pad_step=512)  # (B,C,T)
        return rec.detach()
 
 
class EnCodecView(ViewModel):
    def __init__(self, ckpt: str, device: str, name: str):
        self.name = name; self.device = device
        self.model = CompressionModel.get_pretrained(ckpt, device=device)
        self.model.set_num_codebooks(self.model.total_codebooks)
        self.sr = self.model.sample_rate

        # Check required channels from the underlying model config
        self.channels = 1

        # ---- fp16 on cuda ----
        if device.startswith("cuda"):
            try:
                if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
                    cfg = self.model.model.config
                    self.channels = getattr(cfg, "audio_channels", 1)

                    # ---- ENABLE chunking (lower peak memory) ----
                    cfg.chunk_length_s = None
                    cfg.chunk_stride_s = None
            except Exception as e:
                print(f"[Warn] Could not configure EnCodec {name}: {e}")

        # --- hop length for padding (avoid encode length assertions) ---
        self.frame_rate = getattr(self.model, "frame_rate", None)
        if self.frame_rate is None and hasattr(self.model, "model"):
            self.frame_rate = getattr(self.model.model, "frame_rate", None)
        # fallback: encodec-24k commonly uses 75 fps => hop ~ 320
        if self.frame_rate is None:
            self.hop = 320
        else:
            self.hop = int(round(self.sr / float(self.frame_rate)))
            self.hop = max(1, self.hop)

        # ---- OPTIONAL: fp16 weights on CUDA (works best with autocast below) ----
        if str(device).startswith("cuda"):
            try:
                if hasattr(self.model, "model"):
                    self.model.model = self.model.model.half()
            except Exception as e:
                print(f"[Warn] fp16 failed for {name}: {e}")

    def _ensure_channels(self, w_bct: torch.Tensor) -> torch.Tensor:
        """Force input to match model's expected channel count"""
        B, C, T = w_bct.shape
        if C == self.channels:
            return w_bct
        
        # If model needs stereo (2) but we have mono (1) -> Duplicate
        if self.channels == 2 and C == 1:
            return w_bct.repeat(1, 2, 1)
        
        # If model needs mono (1) but we have stereo (2) -> Mix down
        if self.channels == 1 and C == 2:
            return w_bct.mean(dim=1, keepdim=True)
            
        return w_bct

    def latent(self, w_bct: torch.Tensor) -> torch.Tensor:
        w_bct = as_bct(w_bct).to(self.device)
        w_bct = self._ensure_channels(w_bct)

        m = int(getattr(self, "hop", 320))
        r = w_bct.shape[-1] % m
        if r != 0:
            w_bct = F.pad(w_bct, (0, m - r))

        if str(self.device).startswith("cuda"):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                codes, _ = self.model.encode(w_bct)
                z = self.model.decode_latent(codes)
        else:
            codes, _ = self.model.encode(w_bct)
            z = self.model.decode_latent(codes)
        return z


    @torch.no_grad()
    def codec_roundtrip(self, w_bct: torch.Tensor) -> torch.Tensor:
        w_bct = w_bct.to(self.device)
        w_bct = as_bct(w_bct)                 # ensure (B,C,T)
        w_bct_in = self._ensure_channels(w_bct)

        # padding: after ensuring channels but before encodec chunking (if any)
        if w_bct_in.shape[-1] % 4800 != 0:
            w_bct_in = F.pad(w_bct_in, (0, 4800 - (w_bct_in.shape[-1] % 4800)))

        if self.device.startswith("cuda"):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                codes, _ = self.model.encode(w_bct_in)
                rec = self.model.decode(codes)
        else:
            codes, _ = self.model.encode(w_bct_in)
            rec = self.model.decode(codes)

        # EnCodec decode might return (B,T) or (B,C,T); ensure (B,C,T)
        if w_bct.shape[1] == 1 and rec.shape[1] > 1:
            rec = rec.mean(dim=1, keepdim=True)

        return rec.detach()

class DACView(ViewModel):
    def __init__(self, model_type: str, sr: int, device: str, name: str):
        if dac is None: raise RuntimeError("dac not installed.")
        self.name = name; self.sr = sr; self.device = device
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path).to(device).eval()
        for p in self.model.parameters(): p.requires_grad = False

    def latent(self, w_bct: torch.Tensor) -> torch.Tensor:
        w_bct = w_bct.to(self.device)
        z, codes, latents, *rest = self.model.encode(w_bct)
        if z.dim() == 2: z = z.unsqueeze(0)
        return z 

    @torch.no_grad()
    def codec_roundtrip(self, w_bct: torch.Tensor) -> torch.Tensor:
        w_bct = w_bct.to(self.device)
        # DAC encoder might require length multiple of 4096; if not, pad with zeros at the end
        if w_bct.shape[-1] % 4096 != 0:
             w_bct = F.pad(w_bct, (0, 4096 - (w_bct.shape[-1] % 4096)))
             
        z, codes, latents, *rest = self.model.encode(w_bct)
        rec = self.model.decode(z)
        return rec.detach()

def load_view(name: str, device: str) -> ViewModel:
    """
    name in:
      snac_24, snac_32, snac_44
      encodec_24, encodec_32, encodec_48
      dac_16, dac_24, dac_44
    """
    n = name.lower()
    cache_key = (device, n)
    if cache_key in _VIEW_CACHE:
        return _VIEW_CACHE[cache_key]
    
    # ---- original logic below ----
    if n == "snac_24":
        vm = SNACView("hubertsiuzdak/snac_24khz", 24000, device, "snac_24")
    elif n == "snac_32":
        vm = SNACView("hubertsiuzdak/snac_32khz", 32000, device, "snac_32")
    elif n == "snac_44":
        vm = SNACView("hubertsiuzdak/snac_44khz", 44100, device, "snac_44")
    elif n == "encodec_24":
        vm = EnCodecView("facebook/encodec_24khz", device, "encodec_24")
    elif n == "encodec_32":
        vm = EnCodecView("facebook/encodec_32khz", device, "encodec_32")
    elif n == "encodec_48":
        vm = EnCodecView("facebook/encodec_48khz", device, "encodec_48")
    elif n == "dac_16":
        vm = DACView("16khz", 16000, device, "dac_16")
    elif n == "dac_24":
        vm = DACView("24khz", 24000, device, "dac_24")
    elif n == "dac_44":
        vm = DACView("44khz", 44100, device, "dac_44")
    else:
        raise ValueError(f"Unknown view name: {name}")

    _VIEW_CACHE[cache_key] = vm
    return vm


# --- 1. The Attacker: SNAC ---
from snac import SNAC

class Attack:
    def __init__(self, device):
        self.device = device
        print(f"Loading SNAC on {device}...")
        # SNAC 24kHz
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        self.target_sr = 24000

    def attack(self, audio: torch.Tensor, input_sr: int) -> torch.Tensor:
        with torch.no_grad():
            # normalize input to (B,1,T)
            wav = as_bct(audio)
            wav = ensure_mono(wav)

            # resample to 24k
            wav_24 = torchaudio.functional.resample(wav, input_sr, self.target_sr)

            # encode/decode
            codes = self.model.encode(wav_24.to(self.device))
            rec = self.model.decode(codes)

            # normalize rec to (B,1,T) on cpu
            rec = rec.detach().cpu()
            rec = as_bct(rec)
            rec = ensure_mono(rec)

            # back to input_sr
            out = torchaudio.functional.resample(rec, self.target_sr, input_sr)

            # match original length (use original wav BCT length)
            T0 = wav.shape[-1]
            if out.shape[-1] > T0:
                out = out[..., :T0]
            elif out.shape[-1] < T0:
                out = F.pad(out, (0, T0 - out.shape[-1]))

            return out[0]  # (1,T)

# --- 2. Watermark Implementations ---

class Watermarker:
    def __init__(self, device):
        self.device = device
        self.name = "Base"
    def embed(self, audio, sr): raise NotImplementedError
    def detect(self, audio, sr, payload): raise NotImplementedError

class NoOpWM(Watermarker):
    def __init__(self, device, name="NoOpWM", wm_sr=24000, reason=""):
        super().__init__(device)
        self.name = name
        self.wm_sr = wm_sr
        self.reason = reason
        self.threshold = 0.0
        self.direction = 1.0
        self.calib_info = {"mode": "noop", "reason": reason}

    def embed(self, audio: torch.Tensor, sr: int):
        # Just resample if needed, but do not modify audio
        if sr != self.wm_sr:
            audio = torchaudio.functional.resample(audio, sr, self.wm_sr)
        return audio, None

    def detect(self, audio: torch.Tensor, sr: int, payload):
        return 0.0


class AudioSealWM(Watermarker):
    def __init__(self, device):
        super().__init__(device)
        self.name = "AudioSeal"
        try:
            from audioseal import AudioSeal
            self.generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
            self.detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
            self.wm_sr = 16000
        except ImportError:
            print("AudioSeal not found.")
            self.generator = None

    def embed(self, audio: torch.Tensor, sr: int):
        if self.generator is None: return audio, None
        wav_16k = torchaudio.functional.resample(audio, sr, self.wm_sr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            watermark = self.generator.get_watermark(wav_16k, self.wm_sr)
            watermarked_audio = wav_16k + watermark
        return watermarked_audio.squeeze(0).cpu(), "msg"

    def detect(self, audio: torch.Tensor, sr: int, payload) -> float:
        if self.generator is None: return 0.0
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result, _ = self.detector.detect_watermark(wav_input, self.wm_sr)
            score = result.mean().item() if isinstance(result, torch.Tensor) else result
        return score

class WavMarkWM(Watermarker):
    def __init__(self, device):
        super().__init__(device)
        self.name = "WavMark"
        try:
            import wavmark
            self.model = wavmark.load_model().to(device)
            self.wm_sr = 16000
        except ImportError:
            print("WavMark not found.")
            self.model = None

    def embed(self, audio: torch.Tensor, sr: int):
        if self.model is None: return audio, None
        import wavmark
        wav_16k = torchaudio.functional.resample(audio, sr, self.wm_sr).numpy().flatten()
        payload = np.random.choice([0, 1], size=16)
        try:
            wm_wav, _ = wavmark.encode_watermark(self.model, wav_16k, payload, show_progress=False)
            return torch.tensor(wm_wav).unsqueeze(0), payload
        except: return audio, None

    def detect(self, audio: torch.Tensor, sr: int, payload) -> float:
        if self.model is None: return 0.0
        import wavmark
        if payload is None: return 0.0
        wav_16k = torchaudio.functional.resample(audio, sr, self.wm_sr).numpy().flatten()
        try:
            decoded, _ = wavmark.decode_watermark(self.model, wav_16k, show_progress=False)
            if decoded is None: return 0.0
            return 1.0 - np.mean(payload != decoded) # Accuracy
        except: return 0.0

class SilentCipherWM(Watermarker):
    def __init__(self, device):
        super().__init__(device)
        self.name = "SilentCipher"
        self.wm_sr = 44100 
        self.model = None
        
        try:
            import silentcipher
            # Load model (SilentCipher usually defaults to 44.1k)
            # You might need to update this path or use a default one
            ckpt_path = '../../raw_bench/wm_ckpts/silent_cipher/44_1_khz/73999_iteration'
            config_path = os.path.join(ckpt_path, 'hparams.yaml')
            if os.path.exists(ckpt_path):
                self.model = silentcipher.get_model(
                    ckpt_path=ckpt_path,
                    config_path=config_path,
                    model_type='44.1k', 
                    device=device
                )
                print("[SilentCipher] model loaded.")
            else:
                print("[SilentCipher] Checkpoint not found, skipping.")
        except Exception as e:
            print(f"[SilentCipher] load failed, will be skipped: {e}")
            self.model = None

    def embed(self, audio: torch.Tensor, sr: int):
        if self.model is None: return audio, None
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr)
        wav_np = wav_input.cpu().squeeze().numpy()
        if wav_np.ndim == 0: return audio, None
        if wav_np.ndim > 1: wav_np = wav_np.flatten()
        
        message = [1, 2, 3, 4, 5] 
        try:
            encoded, _ = self.model.encode_wav(wav_np, self.wm_sr, message)
            encoded_tensor = torch.tensor(encoded).to(self.device)
            if encoded_tensor.dim() == 1: encoded_tensor = encoded_tensor.unsqueeze(0)
            return encoded_tensor, message
        except: return audio, None

    def detect(self, audio: torch.Tensor, sr: int, payload) -> float:
        if self.model is None or payload is None: return 0.0
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr)
        wav_np = wav_input.cpu().squeeze().numpy()
        if wav_np.ndim > 1: wav_np = wav_np.flatten()
        try:
            result = self.model.decode_wav(wav_np, self.wm_sr, phase_shift_decoding=False)
            if result is None or 'messages' not in result: return 0.0
            detected_msg = result['messages'][0]
            if detected_msg == payload: return 1.0
            else: return 0.0
        except: return 0.0

class SemanticPCAWM(Watermarker):
    def __init__(self, device):
        super().__init__(device)
        self.name = "SemanticPCA"
        self.wm_sr = 24000 
        snac_vm = load_view("snac_24", device)   # 走 _VIEW_CACHE
        self.model = snac_vm.model               # 共用同一份 weights
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        if hasattr(self.model.quantizer, 'quantizers'):
            self.quantizer_module = self.model.quantizer.quantizers[0]
        else:
            self.quantizer_module = self.model.quantizer
        
        self.codebook = None
        if hasattr(self.quantizer_module, 'codebook'):
             if isinstance(self.quantizer_module.codebook, nn.Embedding):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
        if self.codebook is None:
            for name, module in self.model.quantizer.named_modules():
                if isinstance(module, nn.Embedding):
                    self.codebook = module.weight.detach(); break

        self.projector = None
        for attr in ['in_proj', 'project_in', 'input_conv']:
            if hasattr(self.quantizer_module, attr):
                self.projector = getattr(self.quantizer_module, attr); break
        
        try:
            cb = self.codebook.detach().float().cpu()
            cb_centered = cb - cb.mean(dim=0, keepdim=True)
            _, _, V = torch.linalg.svd(cb_centered, full_matrices=False)
            self.manifold_vector = V[0].unsqueeze(1).to(self.device)
        except Exception as e:
            # fallback: random axis, keep pipeline alive
            dim = int(self.codebook.shape[1])
            v = torch.randn(dim); v = v / (v.norm() + 1e-8)
            self.manifold_vector = v.unsqueeze(1).to(self.device)
            print(f"[WARN][SemanticPCA] SVD failed -> random axis. reason={str(e)[:120]}")

    def get_projected_z(self, audio):
        audio = audio.to(self.device)
        audio = pad_to_multiple(as_bct(audio), 4096)
        audio = ensure_mono(audio)
        z = safe_snac_encoder_out(self.model, audio)  # (B,D,T)
        if self.projector:
            z = self.projector(z)
        return z

    def embed(self, audio, sr):
        epsilon = 0.005; steps = 300; lr = 0.005; target_score = -1.5
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
        if wav_input.dim() < 3: wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
        if wav_input.shape[-1] % 4096 != 0:
            pad = 4096 - (wav_input.shape[-1] % 4096)
            wav_input = torch.nn.functional.pad(wav_input, (0, pad))

        amplitude = wav_input.abs()
        silence_mask = (amplitude > 0.02).float()
        delta = torch.zeros_like(wav_input, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        for i in range(steps):
            optimizer.zero_grad()
            effective_delta = delta * silence_mask
            perturbed = wav_input + effective_delta
            z = self.get_projected_z(perturbed)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            loss = torch.relu(target_score - projections).mean()
            if loss.item() < 1e-4: break 
            loss.backward()
            delta.grad *= silence_mask
            optimizer.step()
            with torch.no_grad(): delta.clamp_(-epsilon, epsilon)

        final_audio = wav_input + (delta.detach() * silence_mask)
        final_audio = final_audio.squeeze().cpu().float()
        if final_audio.dim() == 1: final_audio = final_audio.unsqueeze(0)
        return final_audio, "pca_bit"

    def detect(self, audio, sr, payload):
        with torch.no_grad():
            wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
            if wav_input.dim() < 3: wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
            if wav_input.shape[-1] % 4096 != 0:
                pad = 4096 - (wav_input.shape[-1] % 4096)
                wav_input = torch.nn.functional.pad(wav_input, (0, pad))
            z = self.get_projected_z(wav_input)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            raw_score = projections.mean().item()
            return raw_score

class SemanticClusterWM(Watermarker):
    def __init__(self, device='cuda'):
        super().__init__(device)
        self.name = "SemanticCluster"
        self.wm_sr = 24000 
        snac_vm = load_view("snac_24", device)   # _VIEW_CACHE
        self.model = snac_vm.model               # share weights
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Locate codebook & projector (same logic as PCA)
        if hasattr(self.model.quantizer, 'quantizers'): self.quantizer_module = self.model.quantizer.quantizers[0]
        else: self.quantizer_module = self.model.quantizer
        self.codebook = None
        if hasattr(self.quantizer_module, 'codebook'):
             if isinstance(self.quantizer_module.codebook, nn.Embedding): self.codebook = self.quantizer_module.codebook.weight.detach()
        if self.codebook is None:
            for name, module in self.model.quantizer.named_modules():
                if isinstance(module, nn.Embedding): self.codebook = module.weight.detach(); break
        
        self.projector = None
        for attr in ['in_proj', 'project_in', 'input_conv']:
            if hasattr(self.quantizer_module, attr): self.projector = getattr(self.quantizer_module, attr); break

        # K-Means Axis
        self.manifold_vector = self._compute_kmeans_axis(self.codebook).to(device)

    def _compute_kmeans_axis(self, codebook):
        n_vectors, dim = codebook.shape
        g_cpu = torch.Generator(); g_cpu.manual_seed(42)
        indices = torch.randperm(n_vectors, generator=g_cpu)[:2]
        centroids = codebook[indices].clone()
        for _ in range(10):
            dists = torch.cdist(codebook, centroids)
            labels = torch.argmin(dists, dim=1)
            if labels.sum() == 0 or labels.sum() == n_vectors: break
            centroids[0] = codebook[labels == 0].mean(dim=0)
            centroids[1] = codebook[labels == 1].mean(dim=0)
        vector = centroids[1] - centroids[0]
        vector = vector / (torch.norm(vector) + 1e-8)
        return vector.unsqueeze(1)

    def get_projected_z(self, audio):
        audio = audio.to(self.device)
        audio = pad_to_multiple(as_bct(audio), 4096)
        audio = ensure_mono(audio)
        z = safe_snac_encoder_out(self.model, audio)  # (B,D,T)
        if self.projector:
            z = self.projector(z)
        return z

    def embed(self, audio, sr, target_sdr=32):
        steps = 300; lr = 0.005; target_score = 1.5
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
        if wav_input.dim() < 3: wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
        if wav_input.shape[-1] % 4096 != 0:
            pad = 4096 - (wav_input.shape[-1] % 4096)
            wav_input = torch.nn.functional.pad(wav_input, (0, pad))

        signal_rms = torch.sqrt(torch.mean(wav_input**2))
        epsilon = signal_rms * (10 ** (-target_sdr / 20)) * 2.0
        epsilon = max(1e-4, min(epsilon.item(), 0.1))
        amplitude = wav_input.abs()
        silence_mask = (amplitude > epsilon).float()

        delta = torch.zeros_like(wav_input, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        for i in range(steps):
            optimizer.zero_grad()
            effective_delta = delta * silence_mask
            perturbed = wav_input + effective_delta
            z = self.get_projected_z(perturbed)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            loss = torch.relu(target_score - projections).mean()
            if loss.item() < 1e-4: break 
            loss.backward()
            delta.grad *= silence_mask
            optimizer.step()
            with torch.no_grad(): delta.clamp_(-epsilon, epsilon)

        final_audio = wav_input + (delta.detach() * silence_mask)
        final_audio = final_audio.squeeze().cpu().float()
        if final_audio.dim() == 1: final_audio = final_audio.unsqueeze(0)
        return final_audio, "cluster_bit"

    def detect(self, audio, sr, payload):
        with torch.no_grad():
            wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
            if wav_input.dim() < 3: wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
            if wav_input.shape[-1] % 4096 != 0:
                pad = 4096 - (wav_input.shape[-1] % 4096)
                wav_input = torch.nn.functional.pad(wav_input, (0, pad))
            z = self.get_projected_z(wav_input)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            return projections.mean().item()

class SemanticWM(Watermarker):
    def __init__(self, device='cuda'):
        super().__init__(device)
        self.name = "SemanticRandom"
        self.wm_sr = 24000 

        snac_vm = load_view("snac_24", device)   #  _VIEW_CACHE
        self.model = snac_vm.model               # share weights
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Locate codebook and projector
        if hasattr(self.model.quantizer, 'quantizers'): self.quantizer_module = self.model.quantizer.quantizers[0]
        else: self.quantizer_module = self.model.quantizer
        self.codebook = None
        if hasattr(self.quantizer_module, 'codebook'):
             if isinstance(self.quantizer_module.codebook, nn.Embedding): self.codebook = self.quantizer_module.codebook.weight.detach()
        if self.codebook is None:
             for name, module in self.model.quantizer.named_modules():
                if isinstance(module, nn.Embedding): self.codebook = module.weight.detach(); break
        
        self.projector = None
        for attr in ['in_proj', 'project_in', 'input_conv']:
            if hasattr(self.quantizer_module, attr): self.projector = getattr(self.quantizer_module, attr); break

        latent_dim = self.codebook.shape[1]
        rng = np.random.RandomState(42)
        v_np = rng.randn(latent_dim).astype(np.float32)
        v_np /= np.linalg.norm(v_np)
        self.manifold_vector = torch.tensor(v_np, device=device).unsqueeze(1)

    def get_projected_z(self, audio):
        audio = audio.to(self.device)
        audio = pad_to_multiple(as_bct(audio), 4096)
        audio = ensure_mono(audio)
        z = safe_snac_encoder_out(self.model, audio)  # (B,D,T)
        if self.projector:
            z = self.projector(z)
        return z

    def embed(self, audio, sr, target_sdr=32):
        steps = 300; lr = 0.005; target_score = 1.5
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
        if wav_input.dim() < 3: wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
        if wav_input.shape[-1] % 4096 != 0:
            pad = 4096 - (wav_input.shape[-1] % 4096)
            wav_input = torch.nn.functional.pad(wav_input, (0, pad))

        signal_rms = torch.sqrt(torch.mean(wav_input**2))
        epsilon = signal_rms * (10 ** (-target_sdr / 20)) * 2.0
        epsilon = max(1e-4, min(epsilon.item(), 0.1))
        amplitude = wav_input.abs()
        silence_mask = (amplitude > epsilon).float()

        delta = torch.zeros_like(wav_input, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        for i in range(steps):
            optimizer.zero_grad()
            effective_delta = delta * silence_mask
            perturbed = wav_input + effective_delta
            z = self.get_projected_z(perturbed)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            loss = torch.relu(target_score - projections).mean()
            if loss.item() < 1e-4: break 
            loss.backward()
            delta.grad *= silence_mask
            optimizer.step()
            with torch.no_grad(): delta.clamp_(-epsilon, epsilon)

        final_audio = wav_input + (delta.detach() * silence_mask)
        final_audio = final_audio.squeeze().cpu().float()
        if final_audio.dim() == 1: final_audio = final_audio.unsqueeze(0)
        return final_audio, "random_bit"

    def detect(self, audio, sr, payload):
        with torch.no_grad():
            wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
            if wav_input.dim() < 3: wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
            if wav_input.shape[-1] % 4096 != 0:
                pad = 4096 - (wav_input.shape[-1] % 4096)
                wav_input = torch.nn.functional.pad(wav_input, (0, pad))
            z = self.get_projected_z(wav_input)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            return projections.mean().item()


def build_watermarker(
    wm_name: str,
    device: str,
    opt_codecs: List[str],
    dataset_path: str,
    seed: int,
    calib_files: int,
    fpr: float = 0.01,
    seconds: float = 5.0,
):
    key = wm_name.lower()
    SEMANTIC_KEYS = {"semanticpca", "semanticcluster", "semanticrandom", "semanticwm", "semantic"}


    if key == "jointmanifold":
        wm = JointManifoldWM(
            device=device,
            joint_codecs=opt_codecs,
            wm_sr=24000,
            calib_k=0.5,
            calib_files=calib_files,
            calib_seconds=3.0,
            calib_frames_per_file=512,
            seed=seed,
        )
        try:
            wm.calibrate_from_audio_dir(dataset_path)
        except Exception as e:
            wm.fallback_calibration(reason=str(e))

        wm.threshold = 0.0
        wm.direction = 1.0
        wm.calib_info = {"threshold": 0.0, "direction": 1.0, "rule": "score>0 (normalized margin)"}
        if not hasattr(wm, "calib_info"):
            wm.calib_info = {"mode": "ok"}
        return wm


    if key == "semanticcluster":
        wm = SemanticClusterWM(device=device)
    elif key == "semanticpca":
        wm = SemanticPCAWM(device=device)
    elif key in ("semanticrandom", "semanticwm", "semantic"):
        wm = SemanticWM(device=device)
    else:
        raise ValueError(f"Unknown watermark: {wm_name}")

    
    try:
        info = calibrate_threshold_fpr(
            wm=wm,
            audio_dir=dataset_path,
            calib_files=calib_files,
            seed=seed,
            fpr=fpr,
            seconds=seconds,
        )
        wm.threshold = float(info["threshold"])
        wm.direction = float(info["direction"])
        wm.calib_info = info

        arrow = "score>thr" if wm.direction > 0 else "score<thr"
        print(
            f"[ThreshCalib][{wm.name}] thr={wm.threshold:.4f} ({arrow})  "
            f"FPR(emp)={info['fpr_emp']:.3f} TPR(emp)={info['tpr_emp']:.3f}  N={int(info['N'])}"
        )
    except Exception as e:
        print(f"[WARN][{wm.name}] threshold calib failed -> fallback: {str(e)[:200]}")
        wm.threshold = 0.0
        wm.direction = 1.0
        wm.calib_info = {"mode": "fallback", "reason": str(e)}

    return wm



# ----------------------------
# Attack Router (3 attacks)
# ----------------------------
class AttackRouter:
    def __init__(self, device: str, attack_name: str):
        self.device = device
        self.attack_name = attack_name.lower()
        self.model = load_view(self.attack_name, device)
        self.target_sr = self.model.sr

    @torch.no_grad()
    def attack(self, audio_ct: torch.Tensor, input_sr: int) -> torch.Tensor:
        """
        audio_ct: (C,T) or others
        return: (C,T) at input_sr, length matched to input
        """
        wav = ensure_mono(as_bct(audio_ct))  # (B,1,T)
        T0 = wav.shape[-1]

        w = torchaudio.functional.resample(wav, input_sr, self.target_sr)
        try:
            rec = self.model.codec_roundtrip(w)
        except RuntimeError as e:
            if _is_oom(e):
                # ---- CPU fallback ----
                cuda_cleanup()
                cpu_vm = load_view(self.attack_name, "cpu")
                rec = cpu_vm.codec_roundtrip(w.cpu())
            else:
                raise

        rec = ensure_mono(rec).detach().cpu()
        rec = torchaudio.functional.resample(rec, self.target_sr, input_sr)
        rec = match_length(rec, T0)
        return rec[0]  # (1,T)

# ----------------------------
# JointManifold WM (with detect + pass/fail)
# ----------------------------
class JointManifoldWM:
    def __init__(self, device: str, joint_codecs: list, wm_sr: int = 24000,
                 calib_k: float = 0.5, calib_files: int = 42, 
                 calib_frames_per_file: int = 512,
                 seed: int = 0, **kwargs):
        self.device = device
        self.joint_codecs = [c.lower() for c in joint_codecs]
        self.name = "JointManifold"
        self.wm_sr = int(wm_sr)
        self.calib_k = calib_k
        self.calib_files = calib_files
        self.calib_frames_per_file = int(calib_frames_per_file)
        self.seed = seed
        self.views = {c: load_view(c, device) for c in self.joint_codecs}
        self.v = {}; self.targets = {}; self.scales = {}; self._calibrated = False
        self._init_vectors()

    def _init_vectors(self):
        torch.manual_seed(1234 + self.seed)
        for c, vm in self.views.items():
            dummy = torch.randn(1, 1, vm.sr, device=self.device)
            dummy = pad_to_multiple(dummy, 4096) 
            with torch.no_grad():
                z = vm.latent(dummy)
            D = int(z.shape[1])
            rng = np.random.RandomState(42 + abs(hash(c)) % 10000)
            v = rng.randn(D).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            self.v[c] = torch.tensor(v, device=self.device)

    def _proj(self, z_bdt: torch.Tensor, v_d: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bdt,d->bt", z_bdt, v_d)

    def _prepare_input(self, audio_ct, sr, work_sr=44100):
        wav = torchaudio.functional.resample(audio_ct, sr, work_sr)
        wav = ensure_mono(as_bct(wav)).to(self.device)
        wav = pad_to_multiple(wav, 4096)
        return wav

    @torch.no_grad()
    def calibrate_from_audio_dir(self, audio_dir: str):
        '''
        files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True) + \
                glob.glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True)
        '''
        files = list_audio_files(audio_dir)

        if not files:
            raise RuntimeError(f"[JointManifold] No audio files in {audio_dir}")

        rnd = random.Random(self.seed)
        rnd.shuffle(files)

        # collect samples for each codec
        samples = {c: [] for c in self.joint_codecs}

        # record errors for debugging calibration issues; key=error message, value=count
        err_log = {c: {} for c in self.joint_codecs}
        def _bump_err(c, e):
            msg = (str(e)[:120]).replace("\n", " ")
            err_log[c][msg] = err_log[c].get(msg, 0) + 1

        work_sr = 44100

        # to ensure stable calibration, we require at least min_files with calib_frames_per_file samples each for each codec; if not met, we keep loading more files until we have enough or run out of files
        min_files = 3
        min_samples_per_codec = max(1024, self.calib_frames_per_file * min_files)

        tried = 0
        loaded_ok = 0

        for fp in tqdm(files, desc=f"Calibrating {self.name}", leave=False):
            tried += 1

            # ---- load (CPU) ----
            try:
                wav, sr = cached_load_audio(fp)
            except Exception as e:
                # mp3 backend / corrupted file
                continue

            if wav.numel() == 0:
                continue

            # cap length for speed/stability
            if wav.shape[-1] > sr * 5:
                wav = wav[..., : int(sr * 5)]

            # preprocess on CPU to avoid weird resample/device issues
            try:
                w = torchaudio.functional.resample(wav, sr, work_sr)
                w = ensure_mono(as_bct(w))          # (B,1,T) on CPU
                w = pad_to_multiple(w, 4096)
            except Exception:
                continue

            loaded_ok += 1

            # ---- per-codec extraction ----
            for c, vm in self.views.items():
                try:
                    wv = torchaudio.functional.resample(w, work_sr, vm.sr)  # CPU
                    wv = ensure_mono(as_bct(wv))                            # CPU (B,1,T)
                    # allow views to make correct padding + device move in latent(), to avoid device-specific resampling issues during calibration
                    z = vm.latent(wv)
                    p = self._proj(z, self.v[c])

                    flat = p.flatten()
                    if flat.numel() > self.calib_frames_per_file:
                        idx = torch.randperm(flat.numel(), device=flat.device)[: self.calib_frames_per_file]
                        flat = flat[idx]
                    samples[c].extend(flat.detach().cpu().tolist())

                except Exception as e:
                    _bump_err(c, e)
                    continue

            # if we've already collected enough samples for all codecs, we can stop early to save time
            if all(len(samples[c]) >= min_samples_per_codec for c in self.joint_codecs):
                break

        # ---- sanity check ----
        report = {c: len(samples[c]) for c in self.joint_codecs}
        if not all(report[c] >= min_samples_per_codec for c in self.joint_codecs):
            msg_lines = [
                f"[JointManifold] Calibration failed: samples={report}, required>={min_samples_per_codec}",
                f"  tried_files={tried}, loaded_ok={loaded_ok}",
            ]
            for c in self.joint_codecs:
                if len(samples[c]) < min_samples_per_codec:
                    top = sorted(err_log[c].items(), key=lambda x: -x[1])[:3]
                    if top:
                        msg_lines.append(f"  [{c}] top errors:")
                        for m, k in top:
                            msg_lines.append(f"    x{k}: {m}")
                    else:
                        msg_lines.append(f"  [{c}] no error detail (all skipped early)")
            raise RuntimeError("\n".join(msg_lines))

        # ---- compute targets/scales ----
        for c in self.joint_codecs:
            x = torch.tensor(samples[c], dtype=torch.float32)
            mu = float(x.mean().item())
            sig = float(x.std(unbiased=False).clamp_min(1e-6).item())
            self.targets[c] = mu + self.calib_k * sig
            self.scales[c] = sig

        self._calibrated = True
        print("[Calib] " + " | ".join(
            [f"{c}: N={len(samples[c])}, t={self.targets[c]:.3f}, std={self.scales[c]:.3f}"
            for c in self.joint_codecs]
        ))


    def fallback_calibration(self, reason=""):
        # To make sure embed/detect pipeline is still functional even if calibration fails, we set all targets to 0 and scales to 1, 
        # so that the optimization will try to push the projections to be positive (if direction=1) or negative (if direction=-1) without any normalization. 
        for c in self.joint_codecs:
            self.targets[c] = 0.0
            self.scales[c] = 1.0
        self._calibrated = True
        self.calib_info = {"mode": "fallback", "reason": reason}
        print(f"[WARN][JointManifold] calibration fallback: {reason[:200]}")


    def embed(self, audio_ct: torch.Tensor, sr: int, target_sdr: float = 32.0, steps: int = 300, lr: float = 0.02):
        if not self._calibrated: raise RuntimeError("Calibrate first.")
        
        work_sr = 44100
        # 1. Preprocess
        wav = self._prepare_input(audio_ct, sr, work_sr)
        
        # 2. Compute epsilon based on target SDR and signal RMS
        sig_rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
        eps = float((sig_rms * (10 ** (-target_sdr / 20)) * 2.5).clamp(1e-4, 0.1).item())

        delta = torch.zeros_like(wav, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)

        pbar = tqdm(range(steps), leave=False, desc="Optimizing")
        
        for i in pbar:
            opt.zero_grad()
            pert = wav + delta

            losses = []
            log_loss = {}
            for c, vm in self.views.items():
                wv = torchaudio.functional.resample(pert, work_sr, vm.sr)
                if wv.shape[-1] % 256 != 0: 
                     wv = F.pad(wv, (0, 256 - (wv.shape[-1] % 256)))

                z = vm.latent(wv)
                p = self._proj(z, self.v[c])
                                
                # To avoid std too small view dominating optimization, we do NOT divide by scales[c] here, 
                # so that all views compete in absolute value. 
                # This gives more influence to views with larger std (e.g. dac/snac), which is desirable for the "training" phase balance.
                gap_raw = (self.targets[c] - p) 
                
                # Hinge loss on raw gap
                l_c = torch.relu(gap_raw).pow(2).mean()
                
                norm_factor = 1.0 / (abs(self.targets[c]) + 1e-3)
                l_c = l_c * norm_factor

                losses.append(l_c)
                
                if i % 50 == 0:
                    log_loss[c[:3]] = f"{l_c.item():.2f}"

            loss = torch.stack(losses).mean()
            
            if i % 10 == 0:
                pbar.set_postfix(L=f"{loss.item():.3f}", **log_loss)

            if loss.item() < 1e-4: break
            loss.backward()
            opt.step()
            with torch.no_grad(): delta.clamp_(-eps, eps)

        out = (wav + delta.detach())
        out = torchaudio.functional.resample(out, work_sr, self.wm_sr).detach().cpu()
        return out.squeeze(0), "joint"

    @torch.no_grad()
    def detect(self, audio_ct: torch.Tensor, sr: int, payload=None) -> float:
        if not self._calibrated: raise RuntimeError("Calibrate first.")
        
        work_sr = 44100
        wav = self._prepare_input(audio_ct, sr, work_sr)

        scores = []
        for c, vm in self.views.items():
            wv = torchaudio.functional.resample(wav, work_sr, vm.sr)
            if wv.shape[-1] % 256 != 0:
                 wv = F.pad(wv, (0, 256 - (wv.shape[-1] % 256)))
            
            z = vm.latent(wv)
            p = self._proj(z, self.v[c])
            
            # detect score is the normalized margin to target; positive means "on the right side" of the target, negative means "on the wrong side"
            margin = (p.mean() - self.targets[c]) / self.scales[c]
            scores.append(margin)
        

        # Median is more robust to outliers than mean, which is crucial for the joint manifold where some views might be completely destroyed by attack while others are intact.
        scores_t = torch.stack(scores)
        final_score = float(torch.median(scores_t).item())
        
        return final_score
    
# ----------------------------
# IO / Reporting
# ----------------------------
    
    
def save_artifacts(base_output_dir, dataset_name, exp_tag, filename,
                   orig_wav, wm_wav, attk_wav, sr_orig, sr_wm):
    """
    Save to:
      base_output_dir/
        dataset_name/
          exp_tag/
            <file_stem>/
              1_original.wav
              2_watermarked.wav
              3_attacked.wav
    """
    base = os.path.splitext(filename)[0]
    save_path = os.path.join(base_output_dir, dataset_name, exp_tag, base)
    os.makedirs(save_path, exist_ok=True)

    torchaudio.save(os.path.join(save_path, "1_original.wav"),   orig_wav.cpu(), sr_orig)
    torchaudio.save(os.path.join(save_path, "2_watermarked.wav"), wm_wav.cpu(),  sr_wm)
    torchaudio.save(os.path.join(save_path, "3_attacked.wav"),    attk_wav.cpu(), sr_wm)


def _scalarize(v):
    # Handle standard numpy/torch scalars
    if hasattr(v, 'item') and not isinstance(v, (list, dict, tuple)):
        return v.item()
    
    # Handle numpy arrays specifically
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return v.flatten()[0].item()
        return v.tolist()

    # Handle Torch Tensors
    if torch.is_tensor(v):
        v = v.detach().cpu()
        if v.numel() == 1:
            return v.item()
        return v.numpy().tolist()

    # If it's already a basic type, return it
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v

    # Fallback for complex structures
    try:
        return json.loads(json.dumps(v, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    except:
        return str(v)
    
def _sanitize_row(d):
    # Ensure keys are strings and values are clean primitives
    return {str(k): _scalarize(v) for k, v in d.items()}


def run_one_experiment(
    dataset_path: str,
    base_output_dir: str, 
    out_dir: str,
    opt_name: str,
    opt_codecs: List[str],
    wm_name: str,
    attack_name: str,
    filecount: int,
    calib_files: int = 42,
    save_wavs: bool = True,
    seed: int = 0,
    fpr: float = 0.01,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    exp_tag = f"{opt_name}_{wm_name}_vs_{attack_name}"
    print("\n" + "=" * 72)
    print(f"Exp: {exp_tag}")
    print(f"  Opt codecs : {opt_codecs}")
    print(f"  WM        : {wm_name}")
    print(f"  Attack     : {attack_name}")
    print("=" * 72)

    wm = None
    attacker = None

    try:

        wm = build_watermarker(
            wm_name=wm_name,
            device=device,
            opt_codecs=opt_codecs,
            dataset_path=dataset_path,
            seed=seed,
            calib_files=calib_files,
            fpr=fpr,
            seconds=5.0,
        )
        
        # attacker = AttackRouter(device, attack_name)

        attacker = get_attacker(device, attack_name)


        files = list_audio_files(dataset_path)
        if not files:
            raise RuntimeError(f"No audio files found in {dataset_path}")

        rnd = random.Random(seed)
        rnd.shuffle(files)
        files = files[: min(filecount, len(files))]

        rows = []
        n_pass = 0
        n_fail = 0
        n_err = 0

        for fp in tqdm(files, desc=f"Files {exp_tag}"):
            fname = os.path.basename(fp)
            try:
                wav, sr = cached_load_audio(fp)

                # --- baseline: clean+attack score (only needed for JointManifold delta) ---
                clean_atk_score = None
                if wm_name.lower() == "jointmanifold":
                    wav_wm_sr = torchaudio.functional.resample(wav, sr, wm.wm_sr)   # (1,T) at wm_sr
                    clean_attacked = attacker.attack(wav_wm_sr, wm.wm_sr)           # (1,T) at wm_sr
                    clean_atk_score = float(wm.detect(clean_attacked, wm.wm_sr, payload=None))

                # --- embed with OOM fallback ---
                try:
                    wm_wav, payload = wm.embed(wav, sr)
                except RuntimeError as e:
                    if _is_oom(e) and getattr(wm, "name", "") == "JointManifold":
                        cuda_cleanup()
                        wm_wav, payload = wm.embed(wav, sr, steps=80, target_sdr=24.0, lr=0.02)
                    else:
                        raise
                
                wm_score = float(wm.detect(wm_wav, wm.wm_sr, payload))
                wm_pass  = ((wm_score - float(wm.threshold)) * float(wm.direction) > 0.0)

                if torch.cuda.is_available() and getattr(wm, "name", "") == "JointManifold":
                    torch.cuda.empty_cache()


                attacked = attacker.attack(wm_wav, wm.wm_sr)
                
                atk_score = float(wm.detect(attacked, wm.wm_sr, payload))
                atk_pass  = ((atk_score - float(wm.threshold)) * float(wm.direction) > 0.0)

                clean_wm = torchaudio.functional.resample(wav, sr, wm.wm_sr)
                clean_wm = ensure_mono(clean_wm)  # (1,T)
                # align length for fair clean+attack score comparison; 
                # if JointManifold, this is the same length as wm_wav; 
                # if others, this is the same length as attacked (which is usually slightly shorter than wm_wav due to attack processing)
                T_ref = int(wm_wav.shape[-1])
                clean_wm_bct = match_length(as_bct(clean_wm), T_ref)
                clean_attacked = attacker.attack(clean_wm_bct[0], wm.wm_sr)
                clean_atk_score = float(wm.detect(clean_attacked, wm.wm_sr, payload=None))

                score = atk_score - clean_atk_score
                

                # --- decision ---
                if wm_name.lower() == "jointmanifold" and clean_atk_score is not None:
                    delta = atk_score - clean_atk_score
                    passed = (delta > 0.0)
                    score = float(delta)  
                elif wm.name.lower() in ("semanticcluster","semanticrandom","semanticpca"):
                    passed = (score > 0.0)
                else:
                    atk_pass  = ((atk_score - float(wm.threshold)) * float(wm.direction) > 0.0)
                    passed = atk_pass
                    score = float(atk_score)
                    
                if passed:
                    n_pass += 1
                    result = "PASS"
                else:
                    n_fail += 1
                    result = "FAIL"

                    
                if save_wavs:
                    save_artifacts(base_output_dir, dataset_name, exp_tag, fname, wav, wm_wav, attacked, sr, wm.wm_sr)

                rows.append({
                    "Dataset": os.path.basename(os.path.normpath(dataset_path)),
                    "Opt": opt_name,
                    "WM": wm_name,
                    "Attack": attack_name,
                    "File": fname,
                    "Score": float(score),
                    "Result": result,
                    # debug columns (safe even if None)
                    "WMScore": float(wm_score),
                    "ATKScore": float(atk_score),
                    "CleanATKScore": (None if clean_atk_score is None else float(clean_atk_score)),
                })

            except Exception as e:
                n_err += 1
                rows.append({
                    "Dataset": os.path.basename(os.path.normpath(dataset_path)),
                    "Opt": opt_name,
                    "WM": wm_name,
                    "Attack": attack_name,
                    "File": fname,
                    "Score": None,
                    "Result": "ERROR",
                    "Error": str(e)[:200],
                })
                print(f"\n[ERROR] {exp_tag} on {fname}: {e}")
                traceback.print_exc()

        for i, r in enumerate(rows):
            for k, v in r.items():
                if isinstance(v, np.ndarray) or torch.is_tensor(v):
                    print(f"[BAD ROW] i={i} key={k} type={type(v)} shape={getattr(v,'shape',None)}")
                    break
        
        df = pd.DataFrame([_sanitize_row(r) for r in rows])
        total = max(1, len(df))
        completed = n_pass + n_fail
        pass_rate_completed = n_pass / max(1, completed)   
        pass_over_n         = n_pass / total               
        completed_rate      = completed / total

        stats = {
            "PASS": float(n_pass),
            "FAIL": float(n_fail),
            "ERROR": float(n_err),
            "PASS_rate": float(pass_rate_completed),
            "PASS_over_N": float(pass_over_n),
            "Completed_rate": float(completed_rate),
            "ERROR_rate": float(n_err / total),
            "N": float(len(df)),
        }

        # save per-exp CSV
        exp_csv_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(exp_csv_dir, exist_ok=True)
        df.to_csv(os.path.join(exp_csv_dir, f"results_{exp_tag}.csv"), index=False)

        # print summary
        print("\n[Summary]")
        print(tabulate([[k, v] for k, v in stats.items()], headers=["Metric", "Value"], tablefmt="github"))
        return df, stats

    finally:
        try:
            del wm
            del attacker
        except Exception:
            pass
        cuda_cleanup()

# ----------------------------
# experiments config
# ----------------------------
OPTIMIZATION_SETS = {
    "Opt_A1": ["snac_24", "dac_16", "dac_44"],
    "Opt_A2": ["snac_24", "encodec_24", "encodec_32"],
    "Opt_Mix": ["snac_24", "dac_24", "encodec_24"],
    "Opt_B1": ["snac_32", "dac_16", "dac_44"],
    "Opt_B2": ["snac_32", "encodec_24", "encodec_32"],
}

ATTACK_MODELS = ["snac_44", "encodec_48", "dac_24"]  # 3 attacks

# ----------------------------
# Main
# ----------------------------
def main():
    import argparse

    # default_datasets = ["AIR", "Bach10", "Clotho", "DAPS", "DEMAND", "Freischuetz", "GuitarSet", "jaCappella", "LibriSpeech", "MAESTRO", "PCD"]
    default_datasets = ["LibriSpeech", "Bach10"]


    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=default_datasets, help="dataset folder paths or names")
    parser.add_argument("--skip_datasets", nargs="+", default=[], help="dataset names to skip (by folder name)")
    parser.add_argument("--base_dir", type=str, default="../../dataset", help="base dir for dataset names")
    parser.add_argument("--out", type=str, default="../results_exp15", help="output folder")
    parser.add_argument("--watermarks", nargs="+", default=["JointManifold", "SemanticCluster"], help="Methods to test")
    parser.add_argument("--filecount", type=int, default=120, help="files per experiment per dataset")
    parser.add_argument("--calib_files", type=int, default=42, help="files used for calibration")
    parser.add_argument("--save_wavs", action="store_true", help="save audio artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opt_set", type=str, default="all", choices=list(OPTIMIZATION_SETS.keys()) + ["all"])
    parser.add_argument("--attack", type=str, default="all", choices=ATTACK_MODELS + ["all"])
    parser.add_argument("--fpr", type=float, default=0.01, help="Target FPR for semantic threshold calibration (clean domain).")

    args = parser.parse_args()
    
    base_output_dir = args.out  

    # resolve dataset paths
    dataset_paths = []
    expanded = []
    for d in args.datasets:
        expanded += glob.glob(d)
    ds_list = expanded if expanded else args.datasets

    for d in ds_list:
        if os.path.exists(d) and os.path.isdir(d):
            dataset_paths.append(d)
        else:
            p = os.path.join(args.base_dir, d)
            if os.path.isdir(p):
                dataset_paths.append(p)

    if not dataset_paths:
        raise RuntimeError("No valid dataset folders found.")

    opt_names = list(OPTIMIZATION_SETS.keys()) if args.opt_set == "all" else [args.opt_set]
    atk_names = ATTACK_MODELS if args.attack == "all" else [args.attack]

    # run
    for ds in dataset_paths:
        clear_dataset_audio_cache()  # per-dataset: decode once, reuse within dataset
        ds_name = os.path.basename(os.path.normpath(ds))
        if ds_name in set(args.skip_datasets):
            print(f"[SKIP DATASET] {ds_name}")
            continue
        base_output_dir = "../results_denoised"
        os.makedirs(os.path.join(base_output_dir, ds_name), exist_ok=True)
        
        ds_out_dir = os.path.join(base_output_dir, ds_name)
        os.makedirs(ds_out_dir, exist_ok=True)


        all_stats_rows = []
        print(f"\n\n######## DATASET: {ds_name} ########")
        for wm_name in args.watermarks:
            for opt_name in opt_names:
                clear_model_caches() 
                for atk in atk_names:
                    try:
                        df, stats = run_one_experiment(
                            dataset_path=ds,
                            base_output_dir=base_output_dir,
                            out_dir=ds_out_dir,
                            opt_name=opt_name,
                            opt_codecs=OPTIMIZATION_SETS[opt_name],
                            wm_name=wm_name,
                            attack_name=atk,
                            filecount=args.filecount,
                            calib_files=args.calib_files,
                            save_wavs=args.save_wavs,
                            seed=args.seed,
                            fpr=args.fpr,
                        )
                        all_stats_rows.append({
                            "Dataset": ds_name,
                            "Opt": opt_name,
                            "WM": wm_name,
                            "Attack": atk,
                            **stats
                        })
                    except Exception as e:
                        print(f"[FATAL][SKIP EXP] {ds_name} {wm_name} {opt_name} vs {atk}: {e}")
                        all_stats_rows.append({
                            "Dataset": ds_name,
                            "Opt": opt_name,
                            "WM": wm_name,
                            "Attack": atk,
                            "PASS": 0.0,
                            "FAIL": 0.0,
                            "ERROR": 0.0,
                            "PASS_rate": 0.0,
                            "ERROR_rate": 1.0,
                            "N": 0.0,
                            "FatalError": str(e)[:200],
                        })
                        continue

        for i, r in enumerate(all_stats_rows):
            bad_keys = [k for k in r.keys() if not isinstance(k, str)]
            if bad_keys:
                print(f"[BAD SUMMARY ROW] idx={i} bad_keys_types={[type(k) for k in bad_keys]}")
                print("  bad_keys_preview=", [str(k)[:80] for k in bad_keys])
                print("  row_preview=", {str(k): str(type(v)) for k, v in r.items()})
                break
        
        print(f"DEBUG: Row 0 types: {[(k, type(v)) for k, v in all_stats_rows[0].items()]}")
        
        # dataset-level summary table (15 rows if all)
        summary = pd.DataFrame([_sanitize_row(r) for r in all_stats_rows])
        
        ds_out_dir = os.path.join(base_output_dir, ds_name)
        os.makedirs(ds_out_dir, exist_ok=True)

        summary_path = os.path.join(ds_out_dir, "summary_exp15.csv")
        summary.to_csv(summary_path, index=False)

        # pretty print
        print("\n=== EXP15 SUMMARY (per dataset) ===")
        show_cols = ["WM","Opt","Attack","PASS","FAIL","ERROR","PASS_rate","PASS_over_N","N"]
        print(tabulate(summary[show_cols].values.tolist(), headers=show_cols, tablefmt="github"))
        
        # save summary table and print log into .txt
        log_path = os.path.join(ds_out_dir, "summary_exp15.txt")
        with open(log_path, "w") as f:
            f.write(tabulate(summary[show_cols].values.tolist(), headers=show_cols, tablefmt="github"))

if __name__ == "__main__":
    main()
