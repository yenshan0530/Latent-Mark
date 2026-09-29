import os
import glob
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import torchaudio
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from audiocraft.models import CompressionModel



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    """
    Collapse (channels, time) or (batch, channels, time) to mono.
    """
    if wav.dim() == 2 and wav.size(0) > 1:          # (C, T)
        wav = wav.mean(dim=0, keepdim=True)         # -> (1, T)
    elif wav.dim() == 3 and wav.size(1) > 1:        # (B, C, T)
        wav = wav.mean(dim=1, keepdim=True)         # -> (B, 1, T)
    return wav

def as_bct(wav: torch.Tensor) -> torch.Tensor:
    """
    Force waveform to shape (B, C, T). Accepts:
      (T), (C,T), (B,T), (B,C,T), (B,C,1,T), etc.
    """
    if wav.dim() == 1:
        wav = wav[None, None, :]          # (1,1,T)
    elif wav.dim() == 2:
        # assume (C,T) or (B,T); we treat as (C,T) if first dim small
        if wav.shape[0] <= 2:             # likely channels
            wav = wav[None, :, :]         # (1,C,T)
        else:
            wav = wav[:, None, :]         # (B,1,T)
    elif wav.dim() == 3:
        # already (B,C,T)
        pass
    elif wav.dim() == 4:
        # common bug: (B,C,1,T) -> squeeze the singleton
        if wav.shape[2] == 1:
            wav = wav.squeeze(2)          # (B,C,T)
        else:
            raise RuntimeError(f"Unexpected 4D wav shape: {tuple(wav.shape)}")
    else:
        raise RuntimeError(f"Unexpected wav dim: {wav.dim()} shape={tuple(wav.shape)}")
    return wav


# --- 1. The Attacker: SNAC (Qwen-Omni Tokenizer) ---
from snac import SNAC

class QwenOmniAttack:
    def __init__(self, device):
        self.device = device
        print(f"Loading SNAC (Qwen/Mini-Omni Tokenizer) on {device}...")
        # SNAC 24kHz is the standard for Mini-Omni
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

# ... [Existing Watermarkers: AudioSeal, WavMark, SilentCipher] ...

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
            ckpt_path = '~/spml-final/raw_bench/wm_ckpts/silent_cipher/44_1_khz/73999_iteration'
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

# ... [Existing Semantic methods] ...
class SemanticPCAWM(Watermarker):
    def __init__(self, device):
        super().__init__(device)
        self.name = "SemanticPCA"
        self.wm_sr = 24000 
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        for param in self.model.parameters(): param.requires_grad = False
        
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
        
        cb_centered = self.codebook - self.codebook.mean(dim=0, keepdim=True)
        _, _, V = torch.linalg.svd(cb_centered)
        self.manifold_vector = V[0].unsqueeze(1)

    def get_projected_z(self, audio):
        z = self.model.encoder(audio)[0]
        if z.dim() == 2: z = z.unsqueeze(0)
        if self.projector: z = self.projector(z)
        return z

    def embed(self, audio, sr):
        epsilon = 0.005; steps = 150; lr = 0.005; target_score = -1.5
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
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        for param in self.model.parameters(): param.requires_grad = False
        
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
        z = self.model.encoder(audio)[0]
        if z.dim() == 2: z = z.unsqueeze(0)
        if self.projector: z = self.projector(z)
        return z

    def embed(self, audio, sr, target_sdr=42):
        steps = 150; lr = 0.005; target_score = 1.5
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
    # (Existing Random implementation, abbreviated for brevity as it's similar to Cluster)
    def __init__(self, device='cuda'):
        super().__init__(device)
        self.name = "SemanticRandom"
        self.wm_sr = 24000 
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        for param in self.model.parameters(): param.requires_grad = False
        
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
        z = self.model.encoder(audio)[0]
        if z.dim() == 2: z = z.unsqueeze(0)
        if self.projector: z = self.projector(z)
        return z

    def embed(self, audio, sr, target_sdr=42):
        steps = 150; lr = 0.005; target_score = 1.5
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

# --- 4. NEW: Joint Manifold Optimization (Transferability Method) ---

# class SoundStreamAttack:
#     def __init__(self, device="cpu"):
#         from soundstream import from_pretrained
#         self.codec = from_pretrained().to(device).eval()
#         self.device = device
#         self.target_sr = 16000

#     @torch.no_grad()
#     def attack(self, audio: torch.Tensor, input_sr: int) -> torch.Tensor:
#         # --- normalize input to (B,C,T) ---
#         wav = as_bct(audio)          # (B,C,T)
#         wav = ensure_mono(wav)       # (B,1,T)

#         # --- resample to 16k ---
#         wav16 = torchaudio.functional.resample(wav, input_sr, self.target_sr)  # (B,1,T)

#         # soundstream package expects (B,C,T) float
#         wav16 = wav16.to(self.device)

#         # encode/decode (keep batch dim!)
#         q = self.codec(wav16, mode="encode")
#         rec = self.codec(q, mode="decode")          # expect (B,C,T)

#         # back to cpu for rest of pipeline
#         rec = rec.detach().cpu()

#         # --- resample back to input_sr ---
#         rec = torchaudio.functional.resample(rec, self.target_sr, input_sr)    # (B,1,T)

#         # --- match original length ---
#         T0 = as_bct(audio).shape[-1]
#         if rec.shape[-1] > T0:
#             rec = rec[..., :T0]
#         elif rec.shape[-1] < T0:
#             rec = F.pad(rec, (0, T0 - rec.shape[-1]))

#         # return as (C,T) to match the rest of your framework
#         return rec[0]   # (1,T)


class FunCodecAttack:
    def __init__(self, device="cpu"):
        self.device = device
        self.target_sr = 24000
        try:
            # We attempt to load a standard Speech2Token or native Model
            from funcodec.bin.codec_inference import Speech2Token
            self.codec = Speech2Token("damo/audio_codec-freqcodec_v2-zh_en-24k-16k-32k", device=device)
        except ImportError:
            print("[FunCodecAttack] funcodec not installed or model not found.")
            self.codec = None

    @torch.no_grad()
    def attack(self, audio: torch.Tensor, input_sr: int) -> torch.Tensor:
        wav = as_bct(audio)
        wav = ensure_mono(wav)
        if self.codec is None:
            return wav[0]

        wav_24 = torchaudio.functional.resample(wav, input_sr, self.target_sr).to(self.device)
        
        try:
            # typical FunCodec raw input encoding via model forward or specific encode API
            if hasattr(self.codec, 'encode') and hasattr(self.codec, 'decode'):
                z = self.codec.encode(wav_24)
                rec = self.codec.decode(z)
            elif hasattr(self.codec, 'model'):
                # fallback for modelscope pipeline wrapper
                z, _ = self.codec.model.encode(wav_24)
                rec = self.codec.model.decode(z)
            else:
                rec = wav_24
        except Exception as e:
            print(f"[FunCodecAttack] Inference error: {e}")
            rec = wav_24

        rec = rec.cpu()
        rec = torchaudio.functional.resample(rec, self.target_sr, input_sr)
        
        T0 = wav.shape[-1]
        if rec.shape[-1] > T0:
            rec = rec[..., :T0]
        elif rec.shape[-1] < T0:
            rec = F.pad(rec, (0, T0 - rec.shape[-1]))

        return rec[0]


class APCodecAttack:
    def __init__(self, device="cpu"):
        self.device = device
        self.target_sr = 48000 # APCodec usually operates at 48k or 44.1k
        try:
            import apcodec
            from apcodec.models import APCodecModel 
            # Trying to load a standard from_pretrained definition if it exists
            self.codec = APCodecModel.from_pretrained().to(self.device).eval()
        except (ImportError, AttributeError):
            print("[APCodecAttack] apcodec not installed or bad API structure.")
            self.codec = None

    @torch.no_grad()
    def attack(self, audio: torch.Tensor, input_sr: int) -> torch.Tensor:
        wav = as_bct(audio)
        wav = ensure_mono(wav)
        if self.codec is None: 
            return wav[0]

        wav_sr = torchaudio.functional.resample(wav, input_sr, self.target_sr).to(self.device)
        try:
           z = self.codec.encode(wav_sr)
           rec = self.codec.decode(z)
        except Exception as e:
           print(f"[APCodecAttack] Inference error: {e}")
           rec = wav_sr
           
        rec = torchaudio.functional.resample(rec, self.target_sr, input_sr).cpu()
        T0 = wav.shape[-1]
        if rec.shape[-1] > T0: 
            rec = rec[..., :T0]
        elif rec.shape[-1] < T0: 
            rec = F.pad(rec, (0, T0 - rec.shape[-1]))
        return rec[0]



class AttackRouter:
    """
    Attack choices:
      - "snac"        : QwenOmniAttack (SNAC tokenize/decode)
      - "soundstream" : SoundStreamAttack
      - "funcodec"    : FunCodecAttack
      - "apcodec"     : APCodecAttack
      - "encodec24"   : audiocraft EnCodec 24kHz encode/decode
      - "dac44"       : DAC 44.1kHz encode/decode
      - "encodec32"   : audiocraft EnCodec 32kHz encode/decode

    Default: "snac"
    """
    def __init__(self, device, attack_type="snac"):
        self.device = device
        self.attack_type = attack_type.lower()

        self.snac_attacker = QwenOmniAttack(device) if self.attack_type == "snac" else None
        # self.ss_attacker   = SoundStreamAttack(device=device) if self.attack_type == "soundstream" else None
        self.funcodec_attacker = FunCodecAttack(device=device) if self.attack_type == "funcodec" else None
        self.apcodec_attacker  = APCodecAttack(device=device) if self.attack_type == "apcodec" else None

        # codec attackers (audiocraft)
        self.c_enc24 = None
        self.c_dac44 = None
        self.c_enc32 = None
        if self.attack_type in ("encodec24", "dac44", "encodec32"):
            if self.attack_type == "encodec24":
                self.c_enc24 = CompressionModel.get_pretrained("facebook/encodec_24khz", device=device)
                self.c_enc24.set_num_codebooks(self.c_enc24.total_codebooks)
                self.target_sr = self.c_enc24.sample_rate
            elif self.attack_type == "dac44":
                self.c_dac44 = CompressionModel.get_pretrained("dac_44khz", device=device)
                self.c_dac44.set_num_codebooks(self.c_dac44.total_codebooks)
                self.target_sr = self.c_dac44.sample_rate
            else:
                self.c_enc32 = CompressionModel.get_pretrained("facebook/encodec_32khz", device=device)
                self.c_enc32.set_num_codebooks(self.c_enc32.total_codebooks)
                self.target_sr = self.c_enc32.sample_rate

    @torch.no_grad()
    def attack(self, audio: torch.Tensor, input_sr: int) -> torch.Tensor:
        """
        input audio: (C,T) or (B,C,T) etc.
        return: (C,T) at input_sr (length-matched)
        """
        if self.attack_type == "snac":
            return self.snac_attacker.attack(audio, input_sr)

        if self.attack_type == "soundstream":
            return self.ss_attacker.attack(audio, input_sr)
            
        if self.attack_type == "funcodec":
            return self.funcodec_attacker.attack(audio, input_sr)
            
        if self.attack_type == "apcodec":
            return self.apcodec_attacker.attack(audio, input_sr)

        # codec encode/decode attack
        wav = as_bct(audio)
        wav = ensure_mono(wav)

        # resample to codec sr
        w = torchaudio.functional.resample(wav, input_sr, self.target_sr).to(self.device)

        if self.c_enc24 is not None:
            codes, _ = self.c_enc24.encode(w)
            rec = self.c_enc24.decode(codes)
        elif self.c_dac44 is not None:
            codes, _ = self.c_dac44.encode(w)
            rec = self.c_dac44.decode(codes)
        else:
            codes, _ = self.c_enc32.encode(w)
            rec = self.c_enc32.decode(codes)

        rec = rec.detach().cpu()

        # resample back to input_sr
        rec = torchaudio.functional.resample(rec, self.target_sr, input_sr)

        # match original length
        T0 = as_bct(audio).shape[-1]
        if rec.shape[-1] > T0:
            rec = rec[..., :T0]
        elif rec.shape[-1] < T0:
            rec = F.pad(rec, (0, T0 - rec.shape[-1]))

        return rec[0]  # (1,T)



class JointManifoldWM(Watermarker):
    """
    Joint watermark over a selectable set of 'codec views':
      - 'encodec24' : EnCodec 24kHz latent
      - 'encodec32' : EnCodec 32kHz latent
      - 'dac44'     : DAC 44.1kHz latent
      - 'snac'      : SNAC encoder latent (as watermark space)
      - 'funcodec'  : FunCodec FreqCodec latent
      - 'apcodec'   : APCodec latent

    By default (per your request): ('snac', 'encodec24', 'encodec32')
    """
    def __init__(self, device='cuda', joint_codecs=("snac", "encodec24", "encodec32"),
                 calib_k=1.5, calib_files=42, calib_seconds=3.0, calib_frames_per_file=512):
        super().__init__(device)
        self.name = "JointManifold"
        self.wm_sr = 24000  # framework io sr (what attacker/detector will see)

        # --- codec models ---
        self.c_enc24 = CompressionModel.get_pretrained("facebook/encodec_24khz", device=device)
        self.c_dac44 = CompressionModel.get_pretrained("dac_44khz", device=device)
        self.c_enc32 = CompressionModel.get_pretrained("facebook/encodec_32khz", device=device)

        # ensure deterministic / stable latent stats
        self.c_enc24.set_num_codebooks(self.c_enc24.total_codebooks)
        self.c_dac44.set_num_codebooks(self.c_dac44.total_codebooks)
        self.c_enc32.set_num_codebooks(self.c_enc32.total_codebooks)

        self.sr_enc24 = self.c_enc24.sample_rate   # 24000
        self.sr_dac44 = self.c_dac44.sample_rate   # 44100
        self.sr_enc32 = self.c_enc32.sample_rate   # 32000

        # config
        self.joint_codecs = tuple(joint_codecs)
        self.calib_k = float(calib_k)
        self.calib_files = int(calib_files)
        self.calib_seconds = float(calib_seconds)
        self.calib_frames_per_file = int(calib_frames_per_file)

        # SNAC for joint space
        self.snac = None
        if "snac" in self.joint_codecs:
            self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
            for p in self.snac.parameters():
                p.requires_grad = False
        self.sr_snac = 24000

        self.funcodec = None
        if "funcodec" in self.joint_codecs:
            try:
                from funcodec.bin.codec_inference import Speech2Token
                self.funcodec = Speech2Token("damo/audio_codec-freqcodec_v2-zh_en-24k-16k-32k", device=device)
            except:
                pass
        self.sr_funcodec = 24000

        self.apcodec = None
        if "apcodec" in self.joint_codecs:
            try:
                from apcodec.models import APCodecModel
                self.apcodec = APCodecModel.from_pretrained().to(device).eval()
            except:
                pass
        self.sr_apcodec = 48000

        # per-view direction vectors v[name] : [D,1]
        self.v = {}
        self._init_vectors()

        # filled by calibration
        self.targets = {}  # target[name]
        self.scales  = {}  # scale[name] = baseline gap
        self._calibrated = False

    # ---------- utilities ----------
    def _rand_unit(self, dim: int) -> torch.Tensor:
        rng = np.random.RandomState(42)
        v = rng.randn(dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return torch.tensor(v, device=self.device).unsqueeze(1)  # [D,1]

    def _latent_encodec(self, codec: CompressionModel, wav_bct: torch.Tensor) -> torch.Tensor:
        wav_bct = as_bct(wav_bct)
        wav_bct = ensure_mono(wav_bct)
        codes, _ = codec.encode(wav_bct)
        z = codec.decode_latent(codes)   # [B,D,T]
        return z

    def _latent_snac(self, wav_bct_24k: torch.Tensor) -> torch.Tensor:
        """
        SNAC encoder latent. We use encoder output as continuous z.
        SNAC encoder returns a tuple/list; we take [0].
        """
        wav_bct_24k = as_bct(wav_bct_24k)
        wav_bct_24k = ensure_mono(wav_bct_24k)
        z = self.snac.encoder(wav_bct_24k)[0]  # expect [B,D,T] (or [D,T] -> fix below)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return z

    def _latent_funcodec(self, wav_bct_24k: torch.Tensor) -> torch.Tensor:
        wav_bct_24k = as_bct(wav_bct_24k)
        wav_bct_24k = ensure_mono(wav_bct_24k)
        try:
            if hasattr(self.funcodec, 'model'):
                z, _ = self.funcodec.model.encode(wav_bct_24k)
            else:
                z = self.funcodec.encode(wav_bct_24k)
        except:
            z = torch.zeros((wav_bct_24k.shape[0], 128, wav_bct_24k.shape[-1]//320), device=self.device)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return z

    def _latent_apcodec(self, wav_bct_48k: torch.Tensor) -> torch.Tensor:
        wav_bct_48k = as_bct(wav_bct_48k)
        wav_bct_48k = ensure_mono(wav_bct_48k)
        try:
            z = self.apcodec.encode(wav_bct_48k)
        except:
            z = torch.zeros((wav_bct_48k.shape[0], 128, wav_bct_48k.shape[-1]//320), device=self.device)
        # z usually [B, D, T]
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return z

    def _proj(self, z_bdt: torch.Tensor, v_d1: torch.Tensor) -> torch.Tensor:
        # z: [B,D,T], v: [D,1] -> [B,T]
        return (z_bdt.transpose(1, 2) @ v_d1).squeeze(-1)

    def _hinge(self, proj_bt: torch.Tensor, target: float) -> torch.Tensor:
        return torch.relu(target - proj_bt).mean()

    def _proj_stats(self, proj_bt: torch.Tensor, target: float):
        gap = (target - proj_bt).clamp_min(0.0)
        return (
            float(proj_bt.mean().item()),
            float(proj_bt.std().item()),
            float(proj_bt.min().item()),
            float(proj_bt.max().item()),
            float(gap.mean().item())
        )

    def _init_vectors(self):
        with torch.no_grad():
            # create dummy signals for dimension probing
            dummy_44 = torch.randn(1, 1, self.sr_dac44, device=self.device)  # 1 sec @44.1k
            dummy_24 = torchaudio.functional.resample(dummy_44, self.sr_dac44, self.sr_enc24)
            dummy_32 = torchaudio.functional.resample(dummy_44, self.sr_dac44, self.sr_enc32)
            dummy_sn = torchaudio.functional.resample(dummy_44, self.sr_dac44, self.sr_snac)
            dummy_fc = torchaudio.functional.resample(dummy_44, self.sr_dac44, self.sr_funcodec)
            dummy_ap = torchaudio.functional.resample(dummy_44, self.sr_dac44, self.sr_apcodec)

            # infer dims
            if "encodec24" in self.joint_codecs:
                z = self._latent_encodec(self.c_enc24, dummy_24)
                self.v["encodec24"] = self._rand_unit(z.size(1))
            if "encodec32" in self.joint_codecs:
                z = self._latent_encodec(self.c_enc32, dummy_32)
                self.v["encodec32"] = self._rand_unit(z.size(1))
            if "dac44" in self.joint_codecs:
                z = self._latent_encodec(self.c_dac44, dummy_44)
                self.v["dac44"] = self._rand_unit(z.size(1))
            if "snac" in self.joint_codecs:
                z = self._latent_snac(dummy_sn)
                self.v["snac"] = self._rand_unit(z.size(1))
            if "funcodec" in self.joint_codecs:
                z = self._latent_funcodec(dummy_fc)
                self.v["funcodec"] = self._rand_unit(z.size(1))
            if "apcodec" in self.joint_codecs:
                z = self._latent_apcodec(dummy_ap)
                self.v["apcodec"] = self._rand_unit(z.size(1))

    def _view_latent_and_proj(self, view: str, wav_44k_bct: torch.Tensor):
        """
        wav_44k_bct: [B,1,T] in 44.1k domain
        returns proj: [B,Tv]
        """
        if view == "dac44":
            z = self._latent_encodec(self.c_dac44, wav_44k_bct)
            return self._proj(z, self.v["dac44"])
        elif view == "encodec24":
            w = torchaudio.functional.resample(wav_44k_bct, self.sr_dac44, self.sr_enc24)
            z = self._latent_encodec(self.c_enc24, w)
            return self._proj(z, self.v["encodec24"])
        elif view == "encodec32":
            w = torchaudio.functional.resample(wav_44k_bct, self.sr_dac44, self.sr_enc32)
            z = self._latent_encodec(self.c_enc32, w)
            return self._proj(z, self.v["encodec32"])
        elif view == "snac":
            w = torchaudio.functional.resample(wav_44k_bct, self.sr_dac44, self.sr_snac)
            z = self._latent_snac(w)
            return self._proj(z, self.v["snac"])
        elif view == "funcodec":
            w = torchaudio.functional.resample(wav_44k_bct, self.sr_dac44, self.sr_funcodec)
            z = self._latent_funcodec(w)
            return self._proj(z, self.v["funcodec"])
        elif view == "apcodec":
            w = torchaudio.functional.resample(wav_44k_bct, self.sr_dac44, self.sr_apcodec)
            z = self._latent_apcodec(w)
            return self._proj(z, self.v["apcodec"])
        else:
            raise ValueError(f"Unknown view: {view}")

    @torch.no_grad()
    def calibrate_from_audio_dir(self, audio_dir: str, max_files: int | None = None, seconds: float | None = None):
        import random
        max_files = self.calib_files if max_files is None else int(max_files)
        seconds = self.calib_seconds if seconds is None else float(seconds)

        files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True) + \
                glob.glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True)
        if len(files) == 0:
            raise RuntimeError(f"[JointManifold] No audio files found for calibration in {audio_dir}")

        random.Random(0).shuffle(files)
        files = files[:max_files]

        # frame-level samples per view
        samples = {v: [] for v in self.joint_codecs}

        def _sample_frames(p_bt: torch.Tensor, n: int, seed: int):
            p = p_bt.flatten()
            if p.numel() <= n:
                return p.detach().cpu().tolist()
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            idx = torch.randint(0, p.numel(), (n,), generator=g, device="cpu")
            return p.detach().cpu()[idx].tolist()
        
        def stable_view_id(view: str) -> int:
            # stable across runs
            mapping = {"snac": 1, "encodec24": 2, "encodec32": 3, "dac44": 4, "funcodec": 5, "apcodec": 6}
            return mapping.get(view, 999)



        work_sr = self.sr_dac44  # 44.1k
        good = 0
        for j, fp in enumerate(files):
            try:
                wav, sr = torchaudio.load(fp)
                wav = ensure_mono(wav)  # (1,T)
            except:
                continue

            T = int(sr * seconds)
            if wav.shape[-1] > T:
                wav = wav[..., :T]

            w = torchaudio.functional.resample(wav, sr, work_sr).to(self.device)
            w = as_bct(w); w = ensure_mono(w)

            # align length
            if w.shape[-1] % 4096 != 0:
                pad = 4096 - (w.shape[-1] % 4096)
                w = F.pad(w, (0, pad))

            try:
                for view in self.joint_codecs:
                    proj = self._view_latent_and_proj(view, w)
                    seed = 1234 + 100000 * j + (stable_view_id(view))
                    samples[view] += _sample_frames(proj, n=self.calib_frames_per_file, seed=seed)
                good += 1
            except:
                continue

        if good < 4:
            raise RuntimeError("[JointManifold] Too few readable files for calibration.")

        # compute targets & scales in the SAME unit as hinge
        for view in self.joint_codecs:
            x = torch.tensor(samples[view], dtype=torch.float32)
            mu = float(x.mean().item())
            sig = float(x.std(unbiased=False).clamp_min(1e-6).item())
            t = mu + self.calib_k * sig
            scale = float((t - x).clamp_min(0.0).mean().clamp_min(1e-6).item())
            self.targets[view] = t
            self.scales[view] = scale

        self._calibrated = True
        msg = " | ".join([f"{v}: t={self.targets[v]:.3f}, s={self.scales[v]:.3f}" for v in self.joint_codecs])
        print(f"[JointManifold][Calib] {msg} (k={self.calib_k}, files={good}, frames/file={self.calib_frames_per_file})")

    # ---------- API ----------
    def embed(self, audio: torch.Tensor, sr: int, target_sdr=42):
        if not self._calibrated:
            raise RuntimeError("[JointManifold] Please call calibrate_from_audio_dir(...) before embed().")

        steps = 150
        lr = 0.005

        # optimize in 44.1k domain
        work_sr = self.sr_dac44
        wav = torchaudio.functional.resample(audio, sr, work_sr).to(self.device)
        wav = as_bct(wav); wav = ensure_mono(wav)

        if wav.shape[-1] % 4096 != 0:
            pad = 4096 - (wav.shape[-1] % 4096)
            wav = F.pad(wav, (0, pad))

        # epsilon from SDR
        signal_rms = torch.sqrt(torch.mean(wav ** 2) + 1e-12)
        epsilon = signal_rms * (10 ** (-target_sdr / 20)) * 2.5
        epsilon = float(torch.clamp(epsilon, 1e-4, 0.1).item())

        delta = torch.zeros_like(wav, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)

        log_every = 50
        for i in range(steps):
            opt.zero_grad()
            pert = wav + delta

            # per-view normalized hinge
            losses = {}
            raw_losses = {}
            proj_cache = {}
            for view in self.joint_codecs:
                proj = self._view_latent_and_proj(view, pert)      # [B,Tv]
                proj_cache[view] = proj
                raw = self._hinge(proj, self.targets[view])
                raw_losses[view] = raw
                losses[view] = raw / self.scales[view]

            loss = torch.stack([losses[v] for v in self.joint_codecs]).mean()

            '''
            if (i % log_every == 0) or (i == steps - 1):
                parts = " ".join([f"{v}:{losses[v].item():.3f}(raw{raw_losses[v].item():.3f})" for v in self.joint_codecs])
                print(f"[{i:03d}] L(norm)={loss.item():.4f} | {parts}")
                stat_parts = []
                for v in self.joint_codecs:
                    pm, ps, pmin, pmax, g = self._proj_stats(proj_cache[v], self.targets[v])
                    stat_parts.append(f"{v}: mean={pm:.3f} std={ps:.3f} gap+={g:.3f}")
                print("      " + " | ".join(stat_parts))
            '''

            if loss.item() < 1e-3:
                break

            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)

        final_44 = wav + delta.detach()
        final_24 = torchaudio.functional.resample(final_44, work_sr, self.wm_sr).detach().cpu()
        final_24 = final_24.squeeze(0)  # (1,T) -> (C,T)
        return final_24.float(), "joint_bit"

    def detect(self, audio: torch.Tensor, sr: int, payload) -> float:
        """
        Joint-consistent detection score.
        Return normalized margin average:
          score = mean_view( (mean(proj) - target_view) / scale_view )
        Higher => more confidently above calibrated baseline target.
        """
        if not self._calibrated:
            raise RuntimeError("[JointManifold] calibrate_from_audio_dir(...) before detect().")

        with torch.no_grad():
            work_sr = self.sr_dac44
            w = torchaudio.functional.resample(audio, sr, work_sr).to(self.device)
            w = as_bct(w); w = ensure_mono(w)
            if w.shape[-1] % 4096 != 0:
                pad = 4096 - (w.shape[-1] % 4096)
                w = F.pad(w, (0, pad))

            scores = []
            for view in self.joint_codecs:
                proj = self._view_latent_and_proj(view, w)  # [B,Tv]
                margin = (proj.mean() - self.targets[view]) / self.scales[view]
                scores.append(margin)
            return float(torch.stack(scores).mean().item())

    
    

# --- 5. Main Analysis & Runner ---

def save_artifacts(output_dir, filename, method_name, orig_wav, wm_wav, attk_wav, sr_orig, sr_wm):
    """
    Saves audio files and a comparison plot.
    """
    base_name = os.path.splitext(filename)[0]
    save_path = os.path.join(output_dir, method_name, base_name)
    os.makedirs(save_path, exist_ok=True)

    # Save Audio
    torchaudio.save(os.path.join(save_path, "1_original.wav"), orig_wav.cpu(), sr_orig)
    torchaudio.save(os.path.join(save_path, "2_watermarked.wav"), wm_wav.cpu(), sr_wm)
    torchaudio.save(os.path.join(save_path, "3_lalm_attacked.wav"), attk_wav.cpu(), sr_wm)

    # Plot
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f"Analysis: {method_name} on {filename}", fontsize=16)

    def prep(wav, in_sr, target_sr=16000):
        wav = wav.cpu()
        if wav.dim() > 1: wav = wav.mean(dim=0)
        if in_sr != target_sr:
            wav = torchaudio.functional.resample(wav, in_sr, target_sr)
        return wav.numpy()

    vis_sr = 16000
    w_orig = prep(orig_wav, sr_orig, vis_sr)
    w_wm = prep(wm_wav, sr_wm, vis_sr)
    w_attk = prep(attk_wav, sr_wm, vis_sr)

    min_len = min(len(w_orig), len(w_wm), len(w_attk))
    w_orig, w_wm, w_attk = w_orig[:min_len], w_wm[:min_len], w_attk[:min_len]

    axs[0, 0].plot(w_orig, alpha=0.5, label='Original')
    axs[0, 0].plot(w_wm, alpha=0.5, label='Watermarked', color='orange')
    axs[0, 0].legend()
    axs[0, 0].set_title("Waveform: Original vs Watermarked")
    
    axs[0, 1].plot(w_wm, alpha=0.5, label='Watermarked', color='orange')
    axs[0, 1].plot(w_attk, alpha=0.5, label='LALM Output', color='red')
    axs[0, 1].legend()
    axs[0, 1].set_title("Waveform: Watermarked vs LALM Attack")

    axs[1, 0].specgram(w_wm, Fs=vis_sr, NFFT=1024, noverlap=512, cmap='inferno')
    axs[1, 0].set_title("Spectrogram: Watermarked")
    
    axs[1, 1].specgram(w_attk, Fs=vis_sr, NFFT=1024, noverlap=512, cmap='inferno')
    axs[1, 1].set_title("Spectrogram: LALM Attacked")

    residual = w_wm - w_attk
    axs[2, 0].plot(residual, color='purple', alpha=0.7)
    axs[2, 0].set_title("Residual (Difference)")
    axs[2, 0].set_ylim(-0.1, 0.1) 
    
    axs[2, 1].specgram(residual, Fs=vis_sr, NFFT=1024, noverlap=512, cmap='viridis')
    axs[2, 1].set_title("Residual Spectrogram")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "analysis_plot.png"))
    plt.close(fig)

def find_optimal_threshold(scores, labels):
    """
    scores: list[float]
    labels: list[int] 0/1
    Returns: (best_t, best_acc)
    Candidate thresholds are the midpoints between adjacent unique scores, which is finer than a coarse linspace.
    """
    if len(scores) == 0:
        return 0.5, 0.0

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    uniq = np.unique(scores)
    if len(uniq) == 1:
        # All scores identical; any threshold gives the same result
        t = float(uniq[0])
        preds = (scores > t).astype(np.int32)
        acc = float((preds == labels).mean())
        return t, acc

    # Candidate thresholds = (-inf), midpoints..., (+inf)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate(([-np.inf], mids, [np.inf]))

    best_acc = -1.0
    best_t = float(candidates[0])

    for t in candidates:
        preds = (scores > t).astype(np.int32)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    return best_t, best_acc


def run_qwen_benchmark(audio_dir: str, output_dir: str, watermarks: list[str],
                       filecount: int, thresholds: dict, attack_type: str = "all",
                       joint_codecs: list[str] = ("snac", "encodec24", "encodec32")):
    import traceback

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running Benchmark & Artifact Generation ---")
    os.makedirs(output_dir, exist_ok=True)
    
    if attack_type == "all": # no soundstream
        attack_types = ["snac", "encodec24", "encodec32", "dac44", "funcodec", "apcodec"]
    else:
        attack_types = [attack_type]
    
    attackers = {atype: AttackRouter(device, attack_type=atype) for atype in attack_types}
    

    wm_classes = {
        "AudioSeal": AudioSealWM,
        "WavMark": WavMarkWM,
        "SilentCipher": SilentCipherWM,
        "SemanticPCA": SemanticPCAWM,
        "SemanticCluster": SemanticClusterWM,
        "SemanticRandom": SemanticWM,
        "JointManifold": JointManifoldWM  # <--- Added New Method
    }

    watermarkers = []
    for name in watermarks:
        if name == "JointManifold":
            watermarkers.append(JointManifoldWM(device, joint_codecs=joint_codecs))
        elif name in wm_classes:
            watermarkers.append(wm_classes[name](device))

    files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True) + \
        glob.glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True)
        
    for wm in watermarkers:
        if isinstance(wm, JointManifoldWM):
            wm.calibrate_from_audio_dir(audio_dir, max_files=wm.calib_files, seconds=wm.calib_seconds)


    for atype, attacker in attackers.items():
        results = []
        print(f"\nEvaluating Attack: {atype}")
        for filepath in tqdm(files[:filecount]):
            filename = os.path.basename(filepath)
            try:
                wav, sr = torchaudio.load(filepath)
                wav = ensure_mono(wav) 
                if wav.shape[-1] > sr * 5: wav = wav[:, :sr*5]
            except: continue

            for wm in watermarkers:
                row = {"File": filename, "Method": wm.name, "Survivability": "FAIL", "Score": 0.0}
                try:
                    # 1. Embed
                    wm_audio, payload = wm.embed(wav, sr)
                    current_wm_sr = wm.wm_sr 
                    
                    # 2. Attack (Transferability Test)
                    attacked = attacker.attack(wm_audio, current_wm_sr)
                
                    # 3. Detect
                    score = wm.detect(attacked, current_wm_sr, payload)
                    row["Score"] = round(score, 3)
                    
                    # thresholds is a dict of dicts: {atype: {method: threshold}}
                    attack_thresholds = thresholds.get(atype, {})
                    threshold = attack_thresholds.get(wm.name, 0.5)
                    row["Survivability"] = "PASS" if score > threshold else "FAIL"

                    prefix = "_".join(joint_codecs) if joint_codecs else "default"
                    save_artifacts(output_dir, filename, f"{wm.name}_{prefix}_{atype}", wav, wm_audio, attacked, sr, current_wm_sr)

                except Exception as e:
                    row["Survivability"] = "ERROR"
                    print(f"\n[ERROR] {wm.name} on {filename} with {atype}: {e}")
                    # traceback.print_exc()
                results.append(row)
        
        df = pd.DataFrame(results)
        if not df.empty:
            prefix = "_".join(joint_codecs) if joint_codecs else "default"
            csv_name = f"benchmark_results_{prefix}_{atype}.csv"
            
            print(f"\nSummary for {atype}:")
            # print(df.groupby(["Method", "Survivability"]).size())
            df.to_csv(os.path.join(output_dir, csv_name), index=False)
            
            print(f"Pass rate ({atype}):")
            summary = df.pivot_table(index="Method", columns="Survivability", aggfunc="size", fill_value=0)
            summary["Total"] = summary.sum(axis=1)
            summary["PASS_rate"] = summary.get("PASS", 0) / summary["Total"]
            for col in ["PASS", "FAIL", "ERROR"]:
                if col not in summary:
                    summary[col] = 0
            
            # Reorder columns to ensure consistency
            print(summary[["PASS", "FAIL", "ERROR", "Total", "PASS_rate"]].fillna(0))
            
            # Save the pivot summary table
            summary_csv_name = f"benchmark_summary_{prefix}_{atype}.csv"
            summary.to_csv(os.path.join(output_dir, summary_csv_name), index=True)
    
    
    

def run_detector_checker(audio_dir: str, watermarks: list[str], filecount: int | None,
                         attack_type: str = "snac", threshold_on_attacked: bool = False,
                         joint_codecs: list[str] = ("snac", "encodec24", "encodec32")):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running Detector Checker (NEG/POS + Threshold Estimation) ---")

    if attack_type == "all":
        attack_types = ["snac", "soundstream", "encodec24", "encodec32", "dac44", "funcodec", "apcodec"]
    else:
        attack_types = [attack_type]
    
    attackers = {atype: AttackRouter(device, attack_type=atype) for atype in attack_types}

    wm_classes = {
        "AudioSeal": AudioSealWM, "WavMark": WavMarkWM, "SilentCipher": SilentCipherWM,
        "SemanticPCA": SemanticPCAWM, "SemanticCluster": SemanticClusterWM,
        "SemanticRandom": SemanticWM, "JointManifold": JointManifoldWM
    }
    
    watermarkers = []
    for name in watermarks:
        if name == "JointManifold":
            watermarkers.append(JointManifoldWM(device, joint_codecs=joint_codecs))
        elif name in wm_classes:
            watermarkers.append(wm_classes[name](device))
    
    for wm in watermarkers:
        if isinstance(wm, JointManifoldWM):
            wm.calibrate_from_audio_dir(audio_dir, max_files=wm.calib_files, seconds=wm.calib_seconds)

    # recursive so that subfolders are included
    files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True) + \
            glob.glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True)

    if len(files) == 0:
        print(f"[Warning] No audio files found in: {audio_dir}")
        return {}

    if filecount is None:
        filecount = len(files)
    else:
        filecount = min(filecount, len(files))

    all_thresholds = {}

    for atype, attacker in attackers.items():
        print(f"\n--- Estimating Thresholds for {atype} ---")
        
        # per-method: NEG/POS scores and labels
        score_bank = {wm.name: {"scores": [], "labels": []} for wm in watermarkers}
        results = []

        for filepath in tqdm(files[:filecount]):
            filename = os.path.basename(filepath)
            try:
                wav, sr = torchaudio.load(filepath)
                wav = ensure_mono(wav)  # (1,T)
                if wav.shape[-1] > sr * 5:
                    wav = wav[:, :sr * 5]
            except:
                continue

            for wm in watermarkers:
                # --- NEG: detect on the clean original ---
                try:
                    x = wav
                    x_sr = sr
                    if threshold_on_attacked:
                        x = attacker.attack(x, x_sr)
                    neg_score = wm.detect(x, x_sr, payload=None)
                    score_bank[wm.name]["scores"].append(float(neg_score))
                    score_bank[wm.name]["labels"].append(0)
                    results.append({
                        "File": filename, "Method": wm.name, "Type": "NEG",
                        "Score": round(float(neg_score), 6)
                    })
                except:
                    results.append({
                        "File": filename, "Method": wm.name, "Type": "NEG",
                        "Score": None
                    })

                # --- POS: detect after embedding ---
                try:
                    wm_audio, payload = wm.embed(wav, sr)
                    x = wm_audio
                    x_sr = wm.wm_sr
                    if threshold_on_attacked:
                        x = attacker.attack(x, x_sr)
                    pos_score = wm.detect(x, x_sr, payload)
                    score_bank[wm.name]["scores"].append(float(pos_score))
                    score_bank[wm.name]["labels"].append(1)
                    results.append({
                        "File": filename, "Method": wm.name, "Type": "POS",
                        "Score": round(float(pos_score), 6)
                    })
                except:
                    results.append({
                        "File": filename, "Method": wm.name, "Type": "POS",
                        "Score": None
                    })

        # --- Find the best threshold per method ---
        thresholds = {}
        rows = []
        print(f"\n[Detector Checker] Estimated thresholds for {atype}:")
        for method, d in score_bank.items():
            scores = d["scores"]
            labels = d["labels"]

            if len(scores) == 0:
                t = 0.5
                acc = 0.0
                neg_scores, pos_scores = [], []
            else:
                t, acc = find_optimal_threshold(scores, labels)

                # --- Split NEG/POS scores ---
                neg_scores = [s for s, y in zip(scores, labels) if y == 0]
                pos_scores = [s for s, y in zip(scores, labels) if y == 1]
                
                scores_np = np.asarray(scores, dtype=np.float64)
                labels_np = np.asarray(labels, dtype=np.int32)
                preds = (scores_np > t).astype(np.int32)

                tpr = float((preds[labels_np == 1] == 1).mean()) if np.any(labels_np == 1) else None  # POS pass rate
                tnr = float((preds[labels_np == 0] == 0).mean()) if np.any(labels_np == 0) else None  # NEG pass rate
                fpr = float((preds[labels_np == 0] == 1).mean()) if np.any(labels_np == 0) else None
                fnr = float((preds[labels_np == 1] == 0).mean()) if np.any(labels_np == 1) else None

            thresholds[method] = float(t)

            rows.append({
                "Method": method,
                "Threshold": float(t),
                "Acc": float(acc),
                "N": int(len(scores)),
                "NEG_mean": float(np.mean(neg_scores)) if len(neg_scores) else None,
                "POS_mean": float(np.mean(pos_scores)) if len(pos_scores) else None,
                "TPR": tpr,
                "TNR": tnr,
                "FPR": fpr,
                "FNR": fnr,
            })

            print(f"{method}: t={t:.3f}, acc={acc:.3f}, TPR={tpr:.3f}, TNR={tnr:.3f}, "
                f"N={len(scores)}, "
                f"NEG_mean={np.mean(neg_scores) if len(neg_scores) else 'NA'}, "
                f"POS_mean={np.mean(pos_scores) if len(pos_scores) else 'NA'}")

        # Write per-file details
        df = pd.DataFrame(results)
        prefix = "_".join(joint_codecs) if joint_codecs else "default"
        if not df.empty:
            df.to_csv(os.path.join(audio_dir, f"detector_checker_results_{prefix}_{atype}.csv"), index=False)

        # Write thresholds
        suffix = "attacked" if threshold_on_attacked else "clean"
        pd.DataFrame(rows).to_csv(os.path.join(audio_dir, f"detector_checker_thresholds_{prefix}_{suffix}_{atype}.csv"), index=False)
        
        all_thresholds[atype] = thresholds

    # Returns a dict of dicts: {atype: {method: threshold}} since thresholds vary by attack if threshold_on_attacked=True
    return all_thresholds


if __name__ == "__main__":
    import argparse
    import os
    
    # Default dataset list (used when --datasets is omitted)
    default_datasets = ["AIR", "Bach10", "Clotho", "DAPS", "DEMAND", "Freischuetz", "GuitarSet", "jaCappella", "LibriSpeech", "MAESTRO", "PCD"]
    
    parser = argparse.ArgumentParser()
    
    # Accepts dataset names (resolved under --base_dir) or explicit paths
    parser.add_argument("--datasets", nargs="+", default=default_datasets,
                    help="List of dataset names or full paths (supports glob if expanded by shell)")
    
    parser.add_argument("--watermarks", nargs="+", default=["JointManifold", "SemanticCluster"], help="Methods to test")
    parser.add_argument("--filecount", type=int, default=None,
                    help="How many files to run. Default: run all files in each dataset.")
    parser.add_argument("--mode", choices=["benchmark", "detector", "both"], default="both")
    parser.add_argument("--attack", type=str, default="all",
                    choices=["all", "snac", "soundstream", "funcodec", "apcodec", "encodec24", "dac44", "encodec32"],
                    help="Attack type (default: all)")
    parser.add_argument("--joint_codecs", nargs="+", default=["snac", "encodec24", "encodec32"],
                    help="Codecs to be used by JointManifoldWM as views (default: snac encodec24 encodec32)")
    parser.add_argument("--base_dir", type=str, default="../../dataset",
                    help="Root folder that dataset names are resolved under (default: ../../dataset)")
    parser.add_argument("--out", type=str, default="../results_transferability",
                    help="Output root folder (default: ../results_transferability)")


    args = parser.parse_args()
    
    # Dataset names that are not existing paths are resolved under base_dir
    default_base_dir = args.base_dir
    base_output_dir = args.out
    
    expanded = []
    for d in args.datasets:
        expanded += glob.glob(d)
    args.datasets = expanded if expanded else args.datasets


    for dataset in args.datasets:
        if os.path.isfile(dataset):
            print(f"[Warning] Skipping file path: {dataset}")
            continue

        try:
            # If the argument is an existing path, use it directly
            if os.path.exists(dataset):
                audio_dir = dataset
                # Output folder is named after the last path component
                dataset_name = os.path.basename(os.path.normpath(dataset))
                print(f"[Info] Using custom path: {audio_dir}")
            else:
                # Otherwise treat it as a dataset name under base_dir
                audio_dir = os.path.join(default_base_dir, dataset)
                dataset_name = dataset
            
            # Output path
            output_dir = os.path.join(base_output_dir, dataset_name)
            
            if not os.path.exists(audio_dir):
                print(f"[Warning] Path not found: {audio_dir}")
                continue
            
            # Skip folders with no audio files
            files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True) + glob.glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True)
            if len(files) == 0:
                print(f"[Warning] No .wav or .mp3 files found in {audio_dir}")
                print(f"          If this is a parent folder, please use wildcard like: {audio_dir}/*")
                continue

            print(f"\n=== Dataset: {dataset_name} ===")
            if args.mode == "detector":
                run_detector_checker(audio_dir, args.watermarks, args.filecount, 
                                     attack_type=args.attack,
                                     joint_codecs=args.joint_codecs)
                

            elif args.mode == "benchmark":
                all_thresholds = run_detector_checker(audio_dir, args.watermarks, args.filecount,
                                  attack_type=args.attack, threshold_on_attacked=False,
                                  joint_codecs=args.joint_codecs)
                                  
                # Notice: thresholds differ per attack type since we now iterated them, so the benchmark
                # needs the same `all_thresholds` dict of dicts.
                run_qwen_benchmark(audio_dir, output_dir, args.watermarks, args.filecount,
                   all_thresholds, attack_type=args.attack, joint_codecs=args.joint_codecs)

            elif args.mode == "both":
                all_thresholds = run_detector_checker(audio_dir, args.watermarks, args.filecount,
                                  attack_type=args.attack, threshold_on_attacked=False,
                                  joint_codecs=args.joint_codecs)
                run_qwen_benchmark(audio_dir, output_dir, args.watermarks, args.filecount, all_thresholds, 
                                   attack_type=args.attack, joint_codecs=args.joint_codecs)

        except Exception as e:
            print(f"[Warning] Skipping {dataset}: {e}")
            import traceback
            traceback.print_exc()