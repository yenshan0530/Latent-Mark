"""
Latent-Mark: zero-bit audio watermarking in the latent space of neural audio codecs.

Implements Section 3 of
  "Latent-Mark: An Audio Watermark Robust to Neural Codec Compression" (Interspeech 2026).

Contents
  * codec views      : SNAC / DAC / EnCodec encoders exposing the continuous latent that enters the
                       first residual-vector-quantizer layer, with gradient to the waveform, plus the
                       full encode-quantize-decode round trip used as an attack (Eq. 1).
  * secret axis      : Latent-Cluster (k-means, k=2, Eq. 7), Latent-PCA, Latent-Random.
  * calibration      : null distribution of clean-audio projections -> mu, sigma, tau = mu + k*sigma,
                       alpha = E[ReLU(tau - p_bar)] (Eq. 6, Eq. 8).
  * LatentMark       : single-codec embedding (Eq. 5) and Cross-Codec Optimization (Eq. 9), detection by
                       normalized margin (Eq. 6) with median aggregation over views (Eq. 10).
  * baselines        : AudioSeal, WavMark, SilentCipher wrappers with each method's own detection rule.
  * attacks          : codec round trips and the four DSP attacks of Table 3.
  * io               : audio loading, artifact saving.

Paper defaults (Section 3): gamma = 1.5, k = 1.5, 150 Adam steps, beta = 2.5,
epsilon = clip(beta * RMS(s) * 10^(-SDR/20), 1e-4, 0.1), f_work = 44.1 kHz, T_pad = multiples of 4096.
"""
from __future__ import annotations

import glob
import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


# ----------------------------------------------------------------------------------------------------
# Audio utilities
# ----------------------------------------------------------------------------------------------------
def list_audio_files(folder: str, recursive: bool = False) -> List[str]:
    pattern = os.path.join(folder, "**", "*") if recursive else os.path.join(folder, "*")
    files = [f for f in glob.glob(pattern, recursive=recursive) if f.lower().endswith(AUDIO_EXTS)]
    return sorted(files)


def load_audio(path: str, seconds: Optional[float] = None) -> Tuple[torch.Tensor, int]:
    """Load as mono (1, T) float32. Crops to the first `seconds` if given."""
    wav, sr = torchaudio.load(path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if seconds is not None and wav.shape[-1] > int(sr * seconds):
        wav = wav[:, : int(sr * seconds)]
    return wav.float(), sr


def as_bct(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x[None, None, :]
    if x.dim() == 2:
        return x[None]
    return x


def to_mono(x_bct: torch.Tensor) -> torch.Tensor:
    return x_bct if x_bct.shape[1] == 1 else x_bct.mean(dim=1, keepdim=True)


def resample(x: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return x
    return torchaudio.functional.resample(x, sr_in, sr_out)


def pad_to_multiple(x_bct: torch.Tensor, m: int) -> torch.Tensor:
    r = x_bct.shape[-1] % m
    return x_bct if r == 0 else F.pad(x_bct, (0, m - r))


def match_length(x_bct: torch.Tensor, T: int) -> torch.Tensor:
    if x_bct.shape[-1] > T:
        return x_bct[..., :T]
    if x_bct.shape[-1] < T:
        return F.pad(x_bct, (0, T - x_bct.shape[-1]))
    return x_bct


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


# ----------------------------------------------------------------------------------------------------
# Codec views
# ----------------------------------------------------------------------------------------------------
class CodecView:
    """One neural codec. `latent` returns the continuous representation that enters the first RVQ
    layer (differentiable w.r.t. the waveform); `roundtrip` is the full encode-quantize-decode R(s)."""

    name: str
    sr: int
    hop: int
    device: str

    def codebook(self) -> torch.Tensor:  # (K, D) of the first quantizer layer
        raise NotImplementedError

    def latent(self, x_bct: torch.Tensor) -> torch.Tensor:  # (B, D, T')
        raise NotImplementedError

    @torch.no_grad()
    def roundtrip(self, x_bct: torch.Tensor) -> torch.Tensor:  # (B, 1, T)
        raise NotImplementedError


class SNACView(CodecView):
    HUBS = {24000: "hubertsiuzdak/snac_24khz", 32000: "hubertsiuzdak/snac_32khz", 44100: "hubertsiuzdak/snac_44khz"}

    def __init__(self, sr: int, device: str, name: str):
        from snac import SNAC

        self.name, self.sr, self.device = name, sr, device
        self.model = _freeze(SNAC.from_pretrained(self.HUBS[sr]).to(device))
        self.hop = int(self.model.hop_length)
        self.q0 = self.model.quantizer.quantizers[0]
        self.stride = int(getattr(self.q0, "stride", 1))

    def codebook(self) -> torch.Tensor:
        return self.q0.codebook.weight.detach()

    def latent(self, x_bct: torch.Tensor) -> torch.Tensor:
        # SNAC.preprocess pads to hop * lcm(vq_stride, attention_window), as its own encode() does
        x = self.model.preprocess(to_mono(x_bct).to(self.device))
        z = self.model.encoder(x)
        if self.stride > 1:
            z = F.avg_pool1d(z, self.stride, self.stride)
        return self.q0.in_proj(z)

    @torch.no_grad()
    def roundtrip(self, x_bct: torch.Tensor) -> torch.Tensor:
        x = to_mono(x_bct).to(self.device)
        T = x.shape[-1]
        rec = self.model.decode(self.model.encode(x))
        return match_length(as_bct(rec), T)


class DACView(CodecView):
    TYPES = {16000: "16khz", 24000: "24khz", 44100: "44khz"}

    def __init__(self, sr: int, device: str, name: str):
        import dac

        self.name, self.sr, self.device = name, sr, device
        path = dac.utils.download(model_type=self.TYPES[sr])
        self.model = _freeze(dac.DAC.load(path).to(device))
        self.hop = int(self.model.hop_length)
        self.q0 = self.model.quantizer.quantizers[0]
        self.pad_m = self.hop

    def codebook(self) -> torch.Tensor:
        return self.q0.codebook.weight.detach()

    def latent(self, x_bct: torch.Tensor) -> torch.Tensor:
        x = pad_to_multiple(to_mono(x_bct).to(self.device), self.pad_m)
        return self.q0.in_proj(self.model.encoder(x))

    @torch.no_grad()
    def roundtrip(self, x_bct: torch.Tensor) -> torch.Tensor:
        x = to_mono(x_bct).to(self.device)
        T = x.shape[-1]
        z, *_ = self.model.encode(pad_to_multiple(x, self.pad_m))
        return match_length(as_bct(self.model.decode(z)), T)


class EnCodecView(CodecView):
    CKPTS = {24000: "facebook/encodec_24khz", 32000: "facebook/encodec_32khz", 48000: "facebook/encodec_48khz"}

    def __init__(self, sr: int, device: str, name: str):
        from audiocraft.models import CompressionModel

        self.name, self.sr, self.device = name, sr, device
        self.wrapper = CompressionModel.get_pretrained(self.CKPTS[sr], device=device)
        self.wrapper.set_num_codebooks(self.wrapper.total_codebooks)
        self.inner = _freeze(self.wrapper.model)  # transformers EncodecModel
        self.inner.config.chunk_length_s = None   # encode whole clips (the 48 kHz checkpoint defaults to 1 s chunks)
        self.channels = int(self.wrapper.channels)
        self.hop = int(round(self.sr / float(self.wrapper.frame_rate)))
        self.pad_m = self.hop
        self.normalize = bool(getattr(self.inner.config, "normalize", False))

    def codebook(self) -> torch.Tensor:
        return self.inner.quantizer.layers[0].codebook.embed.detach()

    def _prep(self, x_bct: torch.Tensor) -> torch.Tensor:
        x = pad_to_multiple(to_mono(x_bct).to(self.device), self.pad_m)
        if self.channels == 2:
            x = x.repeat(1, 2, 1)
        return x

    def latent(self, x_bct: torch.Tensor) -> torch.Tensor:
        x = self._prep(x_bct)
        if self.normalize:  # mirrors transformers EncodecModel._encode_frame
            mono = x.mean(dim=1, keepdim=True)
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-5
            x = x / scale
        with torch.backends.cudnn.flags(enabled=False):  # LSTM backward in eval mode
            return self.inner.encoder(x)

    @torch.no_grad()
    def roundtrip(self, x_bct: torch.Tensor) -> torch.Tensor:
        T = x_bct.shape[-1]
        x = self._prep(x_bct)
        bw = self.inner.config.target_bandwidths[-1]  # highest bitrate, all codebooks
        enc = self.inner.encode(x, None, bw)
        rec = self.inner.decode(enc[0], enc[1])[0]
        return match_length(to_mono(as_bct(rec)), T)


VIEW_SPECS: Dict[str, Tuple[type, int]] = {
    "snac_24": (SNACView, 24000), "snac_32": (SNACView, 32000), "snac_44": (SNACView, 44100),
    "dac_16": (DACView, 16000), "dac_24": (DACView, 24000), "dac_44": (DACView, 44100),
    "encodec_24": (EnCodecView, 24000), "encodec_32": (EnCodecView, 32000), "encodec_48": (EnCodecView, 48000),
}
_VIEW_CACHE: Dict[Tuple[str, str], CodecView] = {}


def load_view(name: str, device: str) -> CodecView:
    key = (name.lower(), device)
    if key not in _VIEW_CACHE:
        if key[0] not in VIEW_SPECS:
            raise ValueError(f"Unknown codec view '{name}'. Available: {sorted(VIEW_SPECS)}")
        cls, sr = VIEW_SPECS[key[0]]
        _VIEW_CACHE[key] = cls(sr, device, key[0])
    return _VIEW_CACHE[key]


# ----------------------------------------------------------------------------------------------------
# Secret axis v_c (Section 3.2, "Choice of Shifting Axis")
# ----------------------------------------------------------------------------------------------------
def cluster_axis(codebook: torch.Tensor, seed: int = 0, iters: int = 100) -> torch.Tensor:
    """k-means (k=2) on the codebook; v = (mu_B - mu_A) / ||mu_B - mu_A||  (Eq. 7)."""
    W = codebook.detach().float().cpu()
    g = torch.Generator().manual_seed(seed)
    centroids = W[torch.randperm(W.shape[0], generator=g)[:2]].clone()
    labels = None
    for _ in range(iters):
        new_labels = torch.cdist(W, centroids).argmin(dim=1)
        if labels is not None and torch.equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(2):
            if (labels == j).any():
                centroids[j] = W[labels == j].mean(dim=0)
    v = centroids[1] - centroids[0]
    return v / (v.norm() + 1e-12)


def pca_axis(codebook: torch.Tensor) -> torch.Tensor:
    """First principal component of the centered codebook (SVD)."""
    W = codebook.detach().float().cpu()
    _, _, Vt = torch.linalg.svd(W - W.mean(dim=0, keepdim=True), full_matrices=False)
    v = Vt[0]
    return v / (v.norm() + 1e-12)


def random_axis(dim: int, seed: int = 42) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    v = torch.tensor(rng.randn(dim).astype(np.float32))
    return v / (v.norm() + 1e-12)


def make_axis(kind: str, codebook: torch.Tensor, seed: int = 0) -> torch.Tensor:
    kind = kind.lower()
    if kind == "cluster":
        return cluster_axis(codebook, seed=seed)
    if kind == "pca":
        return pca_axis(codebook)
    if kind == "random":
        return random_axis(codebook.shape[1], seed=42 + seed)
    raise ValueError(f"Unknown axis kind '{kind}' (cluster | pca | random)")


# ----------------------------------------------------------------------------------------------------
# Calibration on clean audio (Eq. 6 and Eq. 8)
# ----------------------------------------------------------------------------------------------------
@dataclass
class ViewCalibration:
    mu: float
    sigma: float
    tau: float
    alpha: float
    n_frames: int
    n_files: int


# ----------------------------------------------------------------------------------------------------
# Watermarkers
# ----------------------------------------------------------------------------------------------------
class Watermarker:
    name: str = "Base"
    wm_sr: int = 16000
    threshold: float = 0.0  # detected iff score > threshold

    def embed(self, audio: torch.Tensor, sr: int):
        raise NotImplementedError

    def detect(self, audio: torch.Tensor, sr: int, payload=None) -> float:
        raise NotImplementedError

    def calibrate(self, files: Sequence[str], seconds: float):
        pass


class LatentMark(Watermarker):
    """Latent-Mark. One view = Latent-Cluster / Latent-PCA / Latent-Random; several views = Latent-Joint.

    Embedding : min_delta  mean_c ReLU(target_c - p_bar_c(s + delta)) / alpha_c   s.t. ||delta||_inf <= eps
                target_c = gamma (Eq. 5, single view) or tau_c (Eq. 9, joint); alpha_c = 1 for a single view.
    Detection : m_c = (p_bar_c - tau_c) / sigma_c per view; score = median_c m_c (Eq. 6, Eq. 10); pass iff > 0.
    """

    threshold = 0.0

    def __init__(
        self,
        views: Sequence[str],
        device: str,
        axis: str = "cluster",
        k: float = 1.5,
        gamma: float = 1.5,
        steps: int = 150,
        lr: float = 5e-3,
        beta: float = 2.5,
        target_sdr: float = 42.0,
        eps_min: float = 1e-4,
        eps_max: float = 0.1,
        work_sr: Optional[int] = None,
        pad_multiple: int = 4096,
        frames_per_file: int = 512,
        seed: int = 0,
        target_mode: Optional[str] = None,
        sigma_level: str = "file",
        hinge: str = "mean",
        name: Optional[str] = None,
    ):
        if not views:
            raise ValueError("LatentMark needs at least one codec view")
        self.device = device
        self.views: List[CodecView] = [load_view(v, device) for v in views]
        self.axis_kind = axis.lower()
        self.k, self.gamma, self.steps, self.lr = float(k), float(gamma), int(steps), float(lr)
        self.beta, self.target_sdr, self.eps_min, self.eps_max = float(beta), float(target_sdr), float(eps_min), float(eps_max)
        self.pad_multiple, self.frames_per_file, self.seed = int(pad_multiple), int(frames_per_file), int(seed)
        self.work_sr = int(work_sr) if work_sr else (self.views[0].sr if len(self.views) == 1 else 44100)
        self.wm_sr = self.work_sr
        # Embedding target per view. "margin" (default): p_bar_c >= tau_c + gamma*sigma_c, i.e. the detection threshold
        # plus the safety margin gamma of Section 3.2 measured in units of the null std; for several views each hinge is
        # divided by alpha_c (Eq. 9). "tau": p_bar_c >= tau_c with no margin. "gamma": p_bar_c >= gamma as an absolute value.
        self.target_mode = (target_mode or "margin").lower()
        self.sigma_level = sigma_level.lower()   # "file": std of p_bar over clean clips; "frame": std of frame projections
        self.hinge = hinge.lower()               # "mean": ReLU(target - p_bar); "frame": mean_t ReLU(target - p_t)
        self.axes: Dict[str, torch.Tensor] = {v.name: make_axis(self.axis_kind, v.codebook(), seed).to(device) for v in self.views}
        self.calib: Dict[str, ViewCalibration] = {}
        if name:
            self.name = name
        elif len(self.views) == 1:
            self.name = {"cluster": "Latent-Cluster", "pca": "Latent-PCA", "random": "Latent-Random"}[self.axis_kind]
        else:
            self.name = "Latent-Joint"

    # ---- projections ----
    def _prepare(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        x = to_mono(as_bct(audio.float())).to(self.device)
        x = resample(x, sr, self.work_sr)
        return pad_to_multiple(x, self.pad_multiple)

    def _proj_frames(self, view: CodecView, x_work: torch.Tensor) -> torch.Tensor:
        """Frame-level projections <z_{c,t}, v_c> -> (B, T')."""
        z = view.latent(resample(x_work, self.work_sr, view.sr))
        return torch.einsum("bdt,d->bt", z, self.axes[view.name])

    def _proj_mean(self, view: CodecView, x_work: torch.Tensor) -> torch.Tensor:
        return self._proj_frames(view, x_work).mean(dim=-1)  # (B,)  p_bar_c (Eq. 4)

    # ---- calibration ----
    @torch.no_grad()
    def calibrate(self, files: Sequence[str], seconds: float = 5.0):
        rng = random.Random(self.seed)
        frames = {v.name: [] for v in self.views}
        gaps: Dict[str, List[float]] = {v.name: [] for v in self.views}
        means: Dict[str, List[float]] = {v.name: [] for v in self.views}
        n_files = 0
        for path in files:
            try:
                wav, sr = load_audio(path, seconds)
            except Exception:
                continue
            x = self._prepare(wav, sr)
            n_files += 1
            for v in self.views:
                p = self._proj_frames(v, x).flatten()
                means[v.name].append(float(p.mean()))
                if p.numel() > self.frames_per_file:
                    idx = torch.tensor(rng.sample(range(p.numel()), self.frames_per_file), device=p.device)
                    p = p[idx]
                frames[v.name].append(p.cpu())
        if n_files == 0:
            raise RuntimeError("Calibration found no readable audio files")
        for v in self.views:
            allf = torch.cat(frames[v.name])
            m = np.asarray(means[v.name], dtype=np.float64)
            mu = float(m.mean())
            if self.sigma_level == "file":
                sigma = float(max(m.std(), 1e-6))
            else:
                sigma = float(allf.std(unbiased=False).clamp_min(1e-6))
            tau = mu + self.k * sigma
            alpha = float(np.mean([max(0.0, tau - m) for m in means[v.name]]))  # Eq. 8
            self.calib[v.name] = ViewCalibration(mu, sigma, tau, max(alpha, 1e-6), int(allf.numel()), n_files)
        print("[calib] " + self.name + " | " + " | ".join(
            f"{n}: mu={c.mu:.3f} sigma={c.sigma:.3f} tau={c.tau:.3f} alpha={c.alpha:.3f}" for n, c in self.calib.items()))
        if self.target_mode == "gamma":
            for n, c in self.calib.items():
                if c.tau >= self.gamma:
                    print(f"[calib][warn] {self.name}/{n}: detection threshold tau={c.tau:.3f} >= gamma={self.gamma}. "
                          f"Raise --gamma or use --target_mode tau.")

    def _require_calib(self):
        if len(self.calib) != len(self.views):
            raise RuntimeError(f"{self.name}: call calibrate(files) before embed/detect")

    # ---- embedding ----
    def embed(self, audio: torch.Tensor, sr: int):
        self._require_calib()
        x = self._prepare(audio, sr)
        T_out = int(math.ceil(audio.shape[-1] * self.work_sr / sr))
        rms = x.pow(2).mean().sqrt().clamp_min(1e-8)
        eps = float((self.beta * rms * 10 ** (-self.target_sdr / 20)).clamp(self.eps_min, self.eps_max))
        delta = torch.zeros_like(x, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            losses = []
            for v in self.views:
                p = self._proj_frames(v, x + delta) if self.hinge == "frame" else self._proj_mean(v, x + delta)
                c = self.calib[v.name]
                if self.target_mode == "gamma":
                    losses.append(F.relu(self.gamma - p).mean())
                elif self.target_mode == "tau":
                    losses.append(F.relu(c.tau - p).mean() / c.alpha)
                else:
                    losses.append(F.relu((c.tau + self.gamma * c.sigma) - p).mean() / c.alpha)
            loss = torch.stack(losses).mean()
            if loss.item() <= 0.0:
                break
            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
        out = (x + delta.detach())[..., :T_out]
        return out[0].cpu(), None

    # ---- detection ----
    @torch.no_grad()
    def margins(self, audio: torch.Tensor, sr: int) -> Dict[str, float]:
        self._require_calib()
        x = self._prepare(audio, sr)
        out = {}
        for v in self.views:
            c = self.calib[v.name]
            out[v.name] = float((self._proj_mean(v, x)[0] - c.tau) / c.sigma)
        return out

    def detect(self, audio: torch.Tensor, sr: int, payload=None) -> float:
        m = list(self.margins(audio, sr).values())
        return float(np.median(m))


# ---- baselines -------------------------------------------------------------------------------------
class AudioSealWM(Watermarker):
    name, wm_sr, threshold = "AudioSeal", 16000, 0.5

    def __init__(self, device: str):
        from audioseal import AudioSeal

        self.device = device
        self.generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device).eval()
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device).eval()

    @torch.no_grad()
    def embed(self, audio, sr):
        x = as_bct(resample(to_mono(as_bct(audio)), sr, self.wm_sr)).to(self.device)
        wm = self.generator.get_watermark(x, self.wm_sr)
        return (x + wm)[0].cpu(), "msg"

    @torch.no_grad()
    def detect(self, audio, sr, payload=None) -> float:
        x = as_bct(resample(to_mono(as_bct(audio)), sr, self.wm_sr)).to(self.device)
        result, _ = self.detector.detect_watermark(x, self.wm_sr)
        return float(result.mean()) if torch.is_tensor(result) else float(result)


class WavMarkWM(Watermarker):
    name, wm_sr, threshold = "WavMark", 16000, 0.85  # bit accuracy over the 16-bit payload

    def __init__(self, device: str, seed: int = 0):
        import wavmark

        self.device = device
        self.model = wavmark.load_model().to(device).eval()
        self.rng = np.random.RandomState(seed)

    MIN_SAMPLES = 17600  # one WavMark chunk (1 s payload window + 10 % shift area) at 16 kHz

    def _prep(self, audio, sr) -> np.ndarray:
        x = resample(to_mono(as_bct(audio)), sr, self.wm_sr)
        if x.shape[-1] < self.MIN_SAMPLES:  # WavMark cannot embed into clips shorter than one chunk
            x = F.pad(x, (0, self.MIN_SAMPLES - x.shape[-1]))
        return x[0, 0].numpy()

    def embed(self, audio, sr):
        import wavmark

        x = self._prep(audio, sr)
        payload = self.rng.choice([0, 1], size=16)
        wm, _ = wavmark.encode_watermark(self.model, x, payload, show_progress=False)
        return torch.tensor(wm, dtype=torch.float32)[None], payload

    def detect(self, audio, sr, payload=None) -> float:
        import wavmark

        x = self._prep(audio, sr)
        decoded, _ = wavmark.decode_watermark(self.model, x, show_progress=False)
        if decoded is None:
            return 0.0
        if payload is None:  # clean audio: score against a random payload
            payload = self.rng.choice([0, 1], size=16)
        return float(1.0 - np.mean(np.asarray(payload) != np.asarray(decoded)))


class SilentCipherWM(Watermarker):
    name, wm_sr, threshold = "SilentCipher", 44100, 0.5  # 1.0 iff the 5-symbol message decodes exactly
    MESSAGE = [1, 2, 3, 4, 5]

    def __init__(self, device: str, ckpt_dir: Optional[str] = None):
        import silentcipher

        self.device = device
        ckpt_dir = ckpt_dir or os.path.join(os.path.dirname(__file__), "..", "..", "raw_bench", "wm_ckpts",
                                            "silent_cipher", "44_1_khz", "73999_iteration")
        # falls back to the Hugging Face download when the local checkpoint is absent
        self.model = silentcipher.get_model(model_type="44.1k", ckpt_path=ckpt_dir,
                                            config_path=os.path.join(ckpt_dir, "hparams.yaml"), device=device)

    def embed(self, audio, sr):
        x = resample(to_mono(as_bct(audio)), sr, self.wm_sr)[0, 0].numpy()
        encoded, _ = self.model.encode_wav(x, self.wm_sr, self.MESSAGE)
        return torch.tensor(np.asarray(encoded), dtype=torch.float32)[None], list(self.MESSAGE)

    def detect(self, audio, sr, payload=None) -> float:
        x = resample(to_mono(as_bct(audio)), sr, self.wm_sr)[0, 0].numpy()
        result = self.model.decode_wav(x, self.wm_sr, phase_shift_decoding=False)
        msgs = (result or {}).get("messages") or []
        return 1.0 if (payload is not None and len(msgs) > 0 and list(msgs[0]) == list(payload)) else 0.0


METHOD_ALIASES = {
    "latent-cluster": "Latent-Cluster", "latentcluster": "Latent-Cluster", "semanticcluster": "Latent-Cluster", "cluster": "Latent-Cluster",
    "latent-pca": "Latent-PCA", "latentpca": "Latent-PCA", "semanticpca": "Latent-PCA", "pca": "Latent-PCA",
    "latent-random": "Latent-Random", "latentrandom": "Latent-Random", "semanticrandom": "Latent-Random", "random": "Latent-Random",
    "latent-joint": "Latent-Joint", "latentjoint": "Latent-Joint", "jointmanifold": "Latent-Joint", "joint": "Latent-Joint",
    "audioseal": "AudioSeal", "wavmark": "WavMark", "silentcipher": "SilentCipher",
}


def canonical_method(name: str) -> str:
    key = name.lower()
    if key not in METHOD_ALIASES:
        raise ValueError(f"Unknown watermark method '{name}'. Choose from {sorted(set(METHOD_ALIASES.values()))}")
    return METHOD_ALIASES[key]


# ----------------------------------------------------------------------------------------------------
# Attacks
# ----------------------------------------------------------------------------------------------------
class Attack:
    name: str

    def __call__(self, audio: torch.Tensor, sr: int) -> torch.Tensor:  # (1, T) -> (1, T) at the same sr
        raise NotImplementedError


class CodecAttack(Attack):
    """Neural codec compression R_a(s) = D(Q(E(s))) through one codec view, resampling in and out."""

    def __init__(self, view_name: str, device: str):
        self.view = load_view(view_name, device)
        self.name = view_name.lower()

    @torch.no_grad()
    def __call__(self, audio, sr):
        x = to_mono(as_bct(audio.float()))
        T = x.shape[-1]
        rec = self.view.roundtrip(resample(x, sr, self.view.sr).to(self.view.device))
        return match_length(resample(rec.cpu(), self.view.sr, sr), T)[0]


class GaussianNoise(Attack):
    def __init__(self, snr_db: float = 60.0, seed: int = 0):
        self.snr_db, self.name = float(snr_db), f"gaussian_{int(snr_db)}dB"
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, audio, sr):
        x = to_mono(as_bct(audio.float()))
        noise_rms = x.pow(2).mean().sqrt() / (10 ** (self.snr_db / 20))
        return (x + torch.randn(x.shape, generator=self.g) * noise_rms)[0]


class AmplitudeScale(Attack):
    def __init__(self, ratio: float = 0.5):
        self.ratio, self.name = float(ratio), f"amplitude_{ratio}"

    def __call__(self, audio, sr):
        return to_mono(as_bct(audio.float()))[0] * self.ratio


class LowPass(Attack):
    def __init__(self, cutoff_hz: float = 4000.0):
        self.cutoff, self.name = float(cutoff_hz), f"lowpass_{int(cutoff_hz)}Hz"

    def __call__(self, audio, sr):
        return torchaudio.functional.lowpass_biquad(to_mono(as_bct(audio.float())), sr, self.cutoff)[0]


class Resample(Attack):
    def __init__(self, target_sr: int = 16000):
        self.target_sr, self.name = int(target_sr), f"resample_{target_sr}Hz"

    def __call__(self, audio, sr):
        x = to_mono(as_bct(audio.float()))
        return match_length(resample(resample(x, sr, self.target_sr), self.target_sr, sr), x.shape[-1])[0]


DSP_ATTACKS = ("gaussian", "amplitude", "lowpass", "resample")
TRANSFER_ATTACKS = ("snac_44", "encodec_48", "dac_24")  # unseen codecs used in Table 2


def make_attack(name: str, device: str, snr_db: float = 60.0, amp_ratio: float = 0.5,
                lowpass_hz: float = 4000.0, resample_hz: int = 16000) -> Attack:
    n = name.lower()
    if n in VIEW_SPECS:
        return CodecAttack(n, device)
    if n == "gaussian":
        return GaussianNoise(snr_db)
    if n == "amplitude":
        return AmplitudeScale(amp_ratio)
    if n == "lowpass":
        return LowPass(lowpass_hz)
    if n == "resample":
        return Resample(resample_hz)
    raise ValueError(f"Unknown attack '{name}'. Codec attacks: {sorted(VIEW_SPECS)}; DSP attacks: {DSP_ATTACKS}")


# ----------------------------------------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------------------------------------
def save_artifacts(out_dir: str, dataset: str, method: str, stem: str,
                   original: torch.Tensor, sr_orig: int,
                   watermarked: torch.Tensor, sr_wm: int,
                   attacked: Optional[torch.Tensor] = None, plot: bool = False):
    """Writes <out>/<dataset>/<method>/<stem>/{1_original,2_watermarked,3_attacked}.wav (+ analysis_plot.png).
    This layout is what audio_quality_check/evaluate_quality.py scans."""
    folder = os.path.join(out_dir, dataset, method, stem)
    os.makedirs(folder, exist_ok=True)
    torchaudio.save(os.path.join(folder, "1_original.wav"), as_bct(original)[0].cpu(), sr_orig)
    torchaudio.save(os.path.join(folder, "2_watermarked.wav"), as_bct(watermarked)[0].cpu(), sr_wm)
    if attacked is not None:
        torchaudio.save(os.path.join(folder, "3_attacked.wav"), as_bct(attacked)[0].cpu(), sr_wm)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis_sr = 16000
        o = resample(as_bct(original)[0].cpu(), sr_orig, vis_sr)[0].numpy()
        w = resample(as_bct(watermarked)[0].cpu(), sr_wm, vis_sr)[0].numpy()
        n = min(len(o), len(w))
        panels = [("Original", o[:n]), ("Watermarked", w[:n]), ("Residual (watermarked - original)", w[:n] - o[:n])]
        if attacked is not None:
            a = resample(as_bct(attacked)[0].cpu(), sr_wm, vis_sr)[0].numpy()
            n = min(n, len(a))
            panels.append(("Attacked", a[:n]))
        fig, axs = plt.subplots(len(panels), 2, figsize=(14, 3 * len(panels)))
        for i, (title, sig) in enumerate(panels):
            axs[i, 0].plot(sig, lw=0.5)
            axs[i, 0].set_title(f"{title}: waveform")
            axs[i, 1].specgram(sig, Fs=vis_sr, NFFT=1024, noverlap=512, cmap="inferno")
            axs[i, 1].set_title(f"{title}: spectrogram")
        fig.suptitle(f"{method} on {stem}")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "analysis_plot.png"), dpi=100)
        plt.close(fig)
