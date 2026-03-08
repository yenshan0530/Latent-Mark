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


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    """
    (channels, time) or (batch, channels, time) -> mono by averaging channels.
    """
    if wav.dim() == 2 and wav.size(0) > 1:          # (C, T)
        wav = wav.mean(dim=0, keepdim=True)         # -> (1, T)
    elif wav.dim() == 3 and wav.size(1) > 1:        # (B, C, T)
        wav = wav.mean(dim=1, keepdim=True)         # -> (B, 1, T)
    return wav

# --- 1. The Attacker: SNAC  ---
from snac import SNAC

class Attack:
    def __init__(self, device):
        self.device = device
        print(f"Loading SNAC on {device}...")
        # SNAC 24kHz is the standard for Mini-Omni
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        self.target_sr = 24000

    def attack(self, audio: torch.Tensor, input_sr: int) -> torch.Tensor:
        """
        Simulates the pipeline:
        Resample -> Quantize to Tokens (SNAC) -> Reconstruct -> Resample back
        """
        with torch.no_grad():
            # 1. Resample to SNAC native rate (24kHz)
            wav_input = torchaudio.functional.resample(audio, input_sr, self.target_sr)
            
            # Ensure shape is (Batch, Channels, Time) -> (1, 1, T)
            if wav_input.dim() == 1: wav_input = wav_input.unsqueeze(0).unsqueeze(0)
            elif wav_input.dim() == 2: wav_input = wav_input.unsqueeze(0)
            
            # 2. Tokenize (The "Kill Zone")
            # SNAC returns a list of codebooks (discrete integers)
            codes = self.model.encode(wav_input.to(self.device))
            
            # 3. Reconstruction (Generation)
            reconstructed_wav = self.model.decode(codes)
            
            # 4. Cleanup & Resample back to original SR for detection
            reconstructed_wav = reconstructed_wav.squeeze().cpu()
            # Handle potential dimension collapse if mono
            # if reconstructed_wav.dim() == 0:
            #     return # Skip if error
            if reconstructed_wav.dim() == 1:
                reconstructed_wav = reconstructed_wav.unsqueeze(0)
                
            out_audio = torchaudio.functional.resample(reconstructed_wav, self.target_sr, input_sr)
            
            # Handl length mismatch (tokenizer often pads/trims slightly)
            if out_audio.shape[-1] > audio.shape[-1]:
                out_audio = out_audio[..., :audio.shape[-1]]
            elif out_audio.shape[-1] < audio.shape[-1]:
                padding = audio.shape[-1] - out_audio.shape[-1]
                out_audio = torch.nn.functional.pad(out_audio, (0, padding))
                
            return out_audio

# --- 2. Watermark Implementations ---

class Watermarker:
    def __init__(self, device):
        self.device = device
        self.name = "Base"
    def embed(self, audio, sr): raise NotImplementedError
    def detect(self, audio, sr, payload): raise NotImplementedError

class AudioSealWM(Watermarker):
    def __init__(self, device):
        super().__init__(device)
        self.name = "AudioSeal"
        from audioseal import AudioSeal
        self.generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
        self.wm_sr = 16000

    def embed(self, audio: torch.Tensor, sr: int):
        wav_16k = torchaudio.functional.resample(audio, sr, self.wm_sr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            watermark = self.generator.get_watermark(wav_16k, self.wm_sr)
            watermarked_audio = wav_16k + watermark
        return watermarked_audio.squeeze(0).cpu(), "msg"

    def detect(self, audio: torch.Tensor, sr: int, payload) -> float:
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result, _ = self.detector.detect_watermark(wav_input, self.wm_sr)
            score = result.mean().item() if isinstance(result, torch.Tensor) else result
        return score

class WavMarkWM(Watermarker):
    def __init__(self, device):
        super().__init__(device)
        self.name = "WavMark"
        import wavmark
        self.model = wavmark.load_model().to(device)
        self.wm_sr = 16000

    def embed(self, audio: torch.Tensor, sr: int):
        import wavmark
        wav_16k = torchaudio.functional.resample(audio, sr, self.wm_sr).numpy().flatten()
        payload = np.random.choice([0, 1], size=16)
        try:
            wm_wav, _ = wavmark.encode_watermark(self.model, wav_16k, payload, show_progress=False)
            return torch.tensor(wm_wav).unsqueeze(0), payload
        except: return audio, None

    def detect(self, audio: torch.Tensor, sr: int, payload) -> float:
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
        import silentcipher

        self.wm_sr = 44100 
        self.model = None 
        
        try:
            # Load model (SilentCipher usually defaults to 44.1k)
            self.model = silentcipher.get_model(
                ckpt_path='../../raw_bench/wm_ckpts/silent_cipher/44_1_khz/73999_iteration',
                config_path='../../raw_bench/wm_ckpts/silent_cipher/44_1_khz/73999_iteration/hparams.yaml',
                model_type='44.1k', 
                device=device
            )
            print("[SilentCipher] model loaded.")
        except Exception as e:
            print(f"[SilentCipher] load failed, will be skipped: {e}")
            self.model = None

    def embed(self, audio: torch.Tensor, sr: int):
        if self.model is None:
            return audio, None
        # 1. Resample to 44.1k
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr)
        
        # 2. Flatten to 1D to avoid (1, 1, T) shape errors
        # SilentCipher prefers a simple (Time,) shape for single file encoding
        wav_np = wav_input.cpu().squeeze().numpy()
        
        # Handle edge case where squeeze removes everything (if scalar) or fails
        if wav_np.ndim == 0: return audio, None
        if wav_np.ndim > 1: wav_np = wav_np.flatten() # Force 1D
        
        # Message
        message = [1, 2, 3, 4, 5] 
        
        try:
            # 3. Encode
            encoded, _ = self.model.encode_wav(wav_np, self.wm_sr, message)
            
            # 4. Convert back to Tensor and ensure (1, T) for consistency with other parts of your pipeline
            encoded_tensor = torch.tensor(encoded).to(self.device)
            
            # Ensure it is (1, T) so torchaudio/attacker can handle it
            if encoded_tensor.dim() == 1: 
                encoded_tensor = encoded_tensor.unsqueeze(0)
            
            return encoded_tensor, message
            
        except Exception as e:
            print(f"SilentCipher Embed Error: {e}")
            return audio, None

    def detect(self, audio: torch.Tensor, sr: int, payload) -> float:
        if self.model is None or payload is None:
            return 0.0
        # 1. Resample
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr)
        
        # 2. Flatten to 1D
        wav_np = wav_input.cpu().squeeze().numpy()
        if wav_np.ndim > 1: wav_np = wav_np.flatten()
        
        try:
            # 3. Decode
            # phase_shift_decoding=False is faster; True is more robust
            result = self.model.decode_wav(wav_np, self.wm_sr, phase_shift_decoding=False)
            
            # Check for valid result structure
            if result is None or 'messages' not in result:
                return 0.0
                
            detected_msg = result['messages'][0]
            
            # 4. Compare (Exact match required for list messages)
            # SilentCipher messages are lists of ints.
            if detected_msg == payload:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            # print(f"SilentCipher Detect Error: {e}")
            return 0.0

class SemanticPCAWM:
    def __init__(self, device):
        self.name = "SemanticPCA"
        self.device = device
        self.wm_sr = 24000 
        
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        for param in self.model.parameters(): param.requires_grad = False
        
        # Auto-Locate
        if hasattr(self.model.quantizer, 'quantizers'):
            self.quantizer_module = self.model.quantizer.quantizers[0]
        else:
            self.quantizer_module = self.model.quantizer
        
        self.codebook = None
        if hasattr(self.quantizer_module, 'codebook'):
             if isinstance(self.quantizer_module.codebook, nn.Embedding):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
             elif hasattr(self.quantizer_module.codebook, 'weight'):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
        
        if self.codebook is None:
            for name, module in self.model.quantizer.named_modules():
                if isinstance(module, nn.Embedding):
                    self.codebook = module.weight.detach()
                    break
        
        # Projector
        self.projector = None
        for attr in ['in_proj', 'project_in', 'input_conv']:
            if hasattr(self.quantizer_module, attr):
                self.projector = getattr(self.quantizer_module, attr)
                break
        
        # PCA Alignment
        cb_centered = self.codebook - self.codebook.mean(dim=0, keepdim=True)
        _, _, V = torch.linalg.svd(cb_centered)
        self.manifold_vector = V[0].unsqueeze(1)

        # --- Visualization of PCA manifold ---
        try:
            import matplotlib.pyplot as plt
            cb_np = self.codebook.cpu().numpy()
            cb_centered = cb_np - cb_np.mean(axis=0, keepdims=True)
            pca_proj = np.dot(cb_centered, V.cpu().numpy().T[:, :2])
            plt.figure(figsize=(6, 6))
            plt.scatter(pca_proj[:, 0], pca_proj[:, 1], s=10, alpha=0.6, label="Codebook Vectors")
            plt.arrow(0, 0, V[0, 0].item(), V[0, 1].item(), color="red", width=0.01, label="Manifold Axis (1st PC)")
            plt.title("PCA Manifold Visualization (SemanticPCAWM)")
            plt.legend()
            os.makedirs("../results/manifold_plots", exist_ok=True)
            plt.savefig("../results/manifold_plots/semantic_pca_manifold.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not draw PCA manifold: {e}")

    def get_projected_z(self, audio):
        z = self.model.encoder(audio)[0]
        if z.dim() == 2: z = z.unsqueeze(0)
        if self.projector: z = self.projector(z)
        return z

    def embed(self, audio, sr):
        # Settings for Imperceptibility
        epsilon = 0.005 
        steps = 150
        lr = 0.005
        # target_score = 3
        target_score = -1.5
        
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
            
            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)

        final_audio = wav_input + (delta.detach() * silence_mask)
        
        # Ensure clean CPU float tensor for saving
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
            # return 1.0 if raw_score > 0.5 else 0.0
            return raw_score

class SemanticClusterWM:
    def __init__(self, device='cuda'):
        self.name = "SemanticCluster"
        self.device = device
        self.wm_sr = 24000 
        
        print(f"Loading SNAC on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        for param in self.model.parameters(): param.requires_grad = False
        
        # --- 1. Locate Codebook ---
        if hasattr(self.model.quantizer, 'quantizers'):
            self.quantizer_module = self.model.quantizer.quantizers[0]
        else:
            self.quantizer_module = self.model.quantizer
        
        self.codebook = None
        if hasattr(self.quantizer_module, 'codebook'):
             if isinstance(self.quantizer_module.codebook, nn.Embedding):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
             elif hasattr(self.quantizer_module.codebook, 'weight'):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
        
        if self.codebook is None:
            for name, module in self.model.quantizer.named_modules():
                if isinstance(module, nn.Embedding):
                    self.codebook = module.weight.detach()
                    break
        
        if self.codebook is None: raise AttributeError("Codebook not found.")

        # --- 2. Locate Projector ---
        self.projector = None
        for attr in ['in_proj', 'project_in', 'input_conv']:
            if hasattr(self.quantizer_module, attr):
                self.projector = getattr(self.quantizer_module, attr)
                break

        # --- 3. K-MEANS MANIFOLD GENERATION ---
        print("Running K-Means (K=2) on Codebook...")
        manifold_vector = self._compute_kmeans_axis(self.codebook)
        self.manifold_vector = manifold_vector.to(device)
        print("Watermark aligned with Cluster Centroids.")

    def _compute_kmeans_axis(self, codebook):
        """
        Runs simple K-Means (K=2) to find two dominant centroids.
        The manifold vector connects these two centroids.
        """
        # Randomly initialize 2 centroids
        n_vectors, dim = codebook.shape
        # Use a fixed seed for reproducibility of the watermark axis
        g_cpu = torch.Generator()
        g_cpu.manual_seed(42)
        indices = torch.randperm(n_vectors, generator=g_cpu)[:2]
        
        centroids = codebook[indices].clone()
        
        # Iterate (10 steps is plenty for this use case)
        for _ in range(10):
            # Calculate distance from every point to both centroids
            dists = torch.cdist(codebook, centroids)
            
            # Assign labels (0 or 1)
            labels = torch.argmin(dists, dim=1)
            
            # Update centroids
            if labels.sum() == 0 or labels.sum() == n_vectors:
                break # Converged or stuck
            
            # Compute new centers
            center_0 = codebook[labels == 0].mean(dim=0)
            center_1 = codebook[labels == 1].mean(dim=0)
            
            centroids[0] = center_0
            centroids[1] = center_1
            
        # The vector is the direction from Centroid 0 to Centroid 1
        vector = centroids[1] - centroids[0]
        # Normalize to unit length
        vector = vector / (torch.norm(vector) + 1e-8)
        # --- Visualization of the manifold ---
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            cb_np = codebook.cpu().numpy()
            pca = PCA(n_components=2)
            cb_2d = pca.fit_transform(cb_np)
            plt.figure(figsize=(6, 6))
            plt.scatter(cb_2d[:, 0], cb_2d[:, 1], s=10, alpha=0.6, label="Codebook Vectors")
            c0, c1 = centroids.cpu().numpy()
            c0_2d, c1_2d = pca.transform([c0, c1])
            plt.scatter([c0_2d[0], c1_2d[0]], [c0_2d[1], c1_2d[1]], color="red", label="Centroids")
            plt.plot([c0_2d[0], c1_2d[0]], [c0_2d[1], c1_2d[1]], color="black", linestyle="--", label="Manifold Axis")
            plt.title("K-Means Manifold Visualization")
            plt.legend()
            os.makedirs("../results/manifold_plots", exist_ok=True)
            plt.savefig("../results/manifold_plots/semantic_cluster_manifold.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not draw manifold: {e}")
        return vector.unsqueeze(1) # (Dim, 1)

    def get_projected_z(self, audio):
        z = self.model.encoder(audio)[0]
        if z.dim() == 2: z = z.unsqueeze(0)
        if self.projector: z = self.projector(z)
        return z

    def embed(self, audio, sr, target_sdr=42):
        """
        Optimizes audio to align with the K-Means axis using dynamic epsilon.
        target_sdr: Desired Signal-to-Distortion Ratio in dB.
        """
        steps = 150
        lr = 0.005
        target_score = 1.5
        
        # 1. Prepare Audio
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
        if wav_input.dim() < 3: 
            wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
        
        # Pad to 4096 stride
        if wav_input.shape[-1] % 4096 != 0:
            pad = 4096 - (wav_input.shape[-1] % 4096)
            wav_input = torch.nn.functional.pad(wav_input, (0, pad))

        # 2. Dynamic Epsilon Calculation (SDR)
        signal_rms = torch.sqrt(torch.mean(wav_input**2))
        epsilon = signal_rms * (10 ** (-target_sdr / 20)) * 2.0
        epsilon = max(1e-4, min(epsilon.item(), 0.1))

        # 3. Create Silence Mask
        amplitude = wav_input.abs()
        silence_mask = (amplitude > epsilon).float()

        delta = torch.zeros_like(wav_input, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        # 4. Optimization Loop
        for i in range(steps):
            optimizer.zero_grad()
            
            effective_delta = delta * silence_mask
            perturbed = wav_input + effective_delta
            
            z = self.get_projected_z(perturbed)
            
            # Project onto K-Means Vector
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            
            # Hinge Loss: We want projection > target_score
            loss = torch.relu(target_score - projections).mean()
            
            if loss.item() < 1e-4: break 
            
            loss.backward()
            
            # Zero out gradient on silence
            delta.grad *= silence_mask
            
            optimizer.step()
            
            # Clip noise
            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)

        final_audio = wav_input + (delta.detach() * silence_mask)
        
        # Clean up for return
        final_audio = final_audio.squeeze().cpu().float()
        if final_audio.dim() == 1: final_audio = final_audio.unsqueeze(0)
        
        return final_audio, "cluster_bit"

    def detect(self, audio, sr, payload):
        with torch.no_grad():
            wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
            if wav_input.dim() < 3: 
                wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
            
            if wav_input.shape[-1] % 4096 != 0:
                pad = 4096 - (wav_input.shape[-1] % 4096)
                wav_input = torch.nn.functional.pad(wav_input, (0, pad))

            z = self.get_projected_z(wav_input)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            
            # Simple average score
            raw_score = projections.mean().item()
            return raw_score
            # return 1.0 if raw_score > 0.5 else 0.0

class SemanticWM:
    def __init__(self, device='cuda'):
        self.name = "SemanticRandom"
        self.device = device
        self.wm_sr = 24000 
        
        print(f"Loading SNAC on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        for param in self.model.parameters(): param.requires_grad = False
        
        # --- 1. Locate Codebook (Necessary for dimension checks) ---
        if hasattr(self.model.quantizer, 'quantizers'):
            self.quantizer_module = self.model.quantizer.quantizers[0]
        else:
            self.quantizer_module = self.model.quantizer
        
        self.codebook = None
        if hasattr(self.quantizer_module, 'codebook'):
             if isinstance(self.quantizer_module.codebook, nn.Embedding):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
             elif hasattr(self.quantizer_module.codebook, 'weight'):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
        
        if self.codebook is None:
            for name, module in self.model.quantizer.named_modules():
                if isinstance(module, nn.Embedding):
                    self.codebook = module.weight.detach()
                    break
        
        if self.codebook is None: raise AttributeError("Codebook not found.")

        # --- 2. Locate Projector ---
        self.projector = None
        for attr in ['in_proj', 'project_in', 'input_conv']:
            if hasattr(self.quantizer_module, attr):
                self.projector = getattr(self.quantizer_module, attr)
                break

        # --- 3. Generate Random Manifold Vector ---
        #The dimension of the PROJECTED latent space (usually 8 for SNAC).
        latent_dim = self.codebook.shape[1]
        
        print(f"Generating Random Manifold Vector (Dim: {latent_dim})...")
        
        # Use fixed seed for reproducibility across runs
        rng = np.random.RandomState(42)
        v_np = rng.randn(latent_dim).astype(np.float32)
        v_np /= np.linalg.norm(v_np) # Normalize to unit length
        
        self.manifold_vector = torch.tensor(v_np, device=device).unsqueeze(1) # (Dim, 1)

    def get_projected_z(self, audio):
        z = self.model.encoder(audio)[0]
        if z.dim() == 2: z = z.unsqueeze(0)
        if self.projector: z = self.projector(z)
        return z

    def embed(self, audio, sr, target_sdr=42):
        """
        Optimizes audio to align with a RANDOM axis using dynamic epsilon.
        """
        steps = 150
        lr = 0.005
        target_score = 1.5
        
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
        if wav_input.dim() < 3: 
            wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
        
        # Pad to 4096 stride
        if wav_input.shape[-1] % 4096 != 0:
            pad = 4096 - (wav_input.shape[-1] % 4096)
            wav_input = torch.nn.functional.pad(wav_input, (0, pad))

        # Dynamic Epsilon Calculation
        signal_rms = torch.sqrt(torch.mean(wav_input**2))
        epsilon = signal_rms * (10 ** (-target_sdr / 20)) * 2.0
        epsilon = max(1e-4, min(epsilon.item(), 0.1))

        # Silence Mask
        amplitude = wav_input.abs()
        silence_mask = (amplitude > epsilon).float()

        delta = torch.zeros_like(wav_input, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        for i in range(steps):
            optimizer.zero_grad()
            
            effective_delta = delta * silence_mask
            perturbed = wav_input + effective_delta
            
            z = self.get_projected_z(perturbed)
            
            # Project onto Random Vector
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            
            loss = torch.relu(target_score - projections).mean()
            
            if loss.item() < 1e-4: break 
            
            loss.backward()
            delta.grad *= silence_mask
            optimizer.step()
            
            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)

        final_audio = wav_input + (delta.detach() * silence_mask)
        final_audio = final_audio.squeeze().cpu().float()
        if final_audio.dim() == 1: final_audio = final_audio.unsqueeze(0)
        
        return final_audio, "random_bit"

    def detect(self, audio, sr, payload):
        with torch.no_grad():
            wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).to(self.device)
            if wav_input.dim() < 3: 
                wav_input = wav_input.unsqueeze(0) if wav_input.dim()==2 else wav_input.unsqueeze(0).unsqueeze(0)
            
            if wav_input.shape[-1] % 4096 != 0:
                pad = 4096 - (wav_input.shape[-1] % 4096)
                wav_input = torch.nn.functional.pad(wav_input, (0, pad))

            z = self.get_projected_z(wav_input)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            
            raw_score = projections.mean().item()
            return raw_score

# --- 3. Main Benchmark Loop ---

def save_artifacts(output_dir, filename, method_name, orig_wav, wm_wav, attk_wav, sr_orig, sr_wm):
    """
    Saves audio files and a comparison plot.
    Handles device movement (GPU -> CPU) automatically.
    """
    # 1. Setup Directory: output/method/file_basename/
    base_name = os.path.splitext(filename)[0]
    save_path = os.path.join(output_dir, method_name, base_name)
    os.makedirs(save_path, exist_ok=True)

    # 2. Save Audio Files
    # Ensure all tensors are on CPU before saving!
    torchaudio.save(os.path.join(save_path, "1_original.wav"), orig_wav.cpu(), sr_orig)
    torchaudio.save(os.path.join(save_path, "2_watermarked.wav"), wm_wav.cpu(), sr_wm)
    torchaudio.save(os.path.join(save_path, "3_lalm_attacked.wav"), attk_wav.cpu(), sr_wm)

    # 3. Generate Analysis Plot
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f"Analysis: {method_name} on {filename}", fontsize=16)

    # Helper to process audio for plotting 
    def prep(wav, in_sr, target_sr=16000):
        wav = wav.cpu() # Move to CPU first
        if wav.dim() > 1: wav = wav.mean(dim=0) # Mix to mono
        # Resample for visual comparison if needed
        if in_sr != target_sr:
            wav = torchaudio.functional.resample(wav, in_sr, target_sr)
        return wav.numpy()

    # Prepare data (Normalize to 16k for consistent X-axis)
    vis_sr = 16000
    w_orig = prep(orig_wav, sr_orig, vis_sr)
    w_wm = prep(wm_wav, sr_wm, vis_sr)
    w_attk = prep(attk_wav, sr_wm, vis_sr)

    # Trim to shortest length to prevent shape mismatch in residual
    min_len = min(len(w_orig), len(w_wm), len(w_attk))
    w_orig, w_wm, w_attk = w_orig[:min_len], w_wm[:min_len], w_attk[:min_len]

    # --- Row 1: Time Domain (Waveforms) ---
    axs[0, 0].plot(w_orig, alpha=0.5, label='Original')
    axs[0, 0].plot(w_wm, alpha=0.5, label='Watermarked', color='orange')
    axs[0, 0].legend()
    axs[0, 0].set_title("Waveform: Original vs Watermarked")
    
    axs[0, 1].plot(w_wm, alpha=0.5, label='Watermarked', color='orange')
    axs[0, 1].plot(w_attk, alpha=0.5, label='LALM Output', color='red')
    axs[0, 1].legend()
    axs[0, 1].set_title("Waveform: Watermarked vs LALM Attack")

    # --- Row 2: Frequency Domain (Spectrograms) ---
    axs[1, 0].specgram(w_wm, Fs=vis_sr, NFFT=1024, noverlap=512, cmap='inferno')
    axs[1, 0].set_title("Spectrogram: Watermarked Audio")
    
    axs[1, 1].specgram(w_attk, Fs=vis_sr, NFFT=1024, noverlap=512, cmap='inferno')
    axs[1, 1].set_title("Spectrogram: LALM Attacked Audio")

    # --- Row 3: The "Kill Zone" (Residuals) ---
    residual = w_wm - w_attk
    axs[2, 0].plot(residual, color='purple', alpha=0.7)
    axs[2, 0].set_title("Residual Waveform (Lost Information)")
    axs[2, 0].set_ylim(-0.1, 0.1) 
    
    axs[2, 1].specgram(residual, Fs=vis_sr, NFFT=1024, noverlap=512, cmap='viridis')
    axs[2, 1].set_title("Residual Spectrogram (Where the Watermark Died)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "analysis_plot.png"))
    plt.close(fig)

# --- UPDATED Benchmark Loop ---
def find_optimal_threshold(scores, labels, num_points=100):
    """
    Finds the threshold that maximizes classification accuracy.
    scores: list of float detection scores
    labels: list of int (1 for PASS, 0 for FAIL ground truth)
    """
    if len(scores) == 0:
        return 0.5, 0.0
    thresholds = np.linspace(min(scores), max(scores), num_points)
    best_acc, best_t = 0.0, 0.5
    for t in thresholds:
        preds = [1 if s > t else 0 for s in scores]
        acc = np.mean([p == l for p, l in zip(preds, labels)])
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def run_benchmark(audio_dir: str, output_dir: str, watermarks: list[str], filecount: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running SNAC Benchmark & Artifact Generation ---")

    os.makedirs(output_dir, exist_ok=True)

    attacker = Attack(device)
    # Map available watermarkers
    wm_classes = {
        "AudioSeal": AudioSealWM,
        "WavMark": WavMarkWM,
        "SilentCipher": SilentCipherWM,
        "SemanticPCA": SemanticPCAWM,
        "SemanticCluster": SemanticClusterWM,
        "SemanticRandom": SemanticWM
    }

    # Filter based on user input
    watermarkers = [wm_classes[name](device) for name in watermarks if name in wm_classes]

    files = glob.glob(os.path.join(audio_dir, "*.wav")) + glob.glob(os.path.join(audio_dir, "*.mp3"))
    results = []

    filecount = min(filecount, len(files))
    print(f"Found {len(files)} files. Processing first {filecount}...")

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
                
                # 2. Attack
                attacked_audio = attacker.attack(wm_audio, current_wm_sr)

                # 3. Detect
                score = wm.detect(attacked_audio, current_wm_sr, payload)
                row["Score"] = round(score, 3)
                
                # Thresholds
                if wm.name == "AudioSeal": threshold = 0.5 
                elif wm.name == "SilentCipher": threshold = 0.99
                else: threshold = 0.85
                row["Survivability"] = "PASS" if score > threshold else "FAIL"

                # 4. SAVE ARTIFACTS (Audio + Plots)
                save_artifacts(
                    output_dir, filename, wm.name,
                    wav, wm_audio, attacked_audio,
                    sr, current_wm_sr
                )
                
            except Exception as e:
                row["Survivability"] = "ERROR"
                print(f"Error in {wm.name} on {filename}: {e}") 

            results.append(row)
    
    # Report
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    if not df.empty:
        # print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        print("\nSummary:")
        print(df.groupby(["Method", "Survivability"]).size())
        print(f"\nArtifacts saved to: {os.path.abspath(output_dir)}")

        # --- Save summary to file ---
        summary_path = os.path.join(output_dir, "benchmark_summary.txt")
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        with open(summary_path, "w") as f:
            f.write("Benchmark Summary (Survivability)\n")
            f.write("="*60 + "\n")
            f.write(str(df.groupby(["Method", "Survivability"]).size()))
            f.write("\n\nArtifacts saved to: " + os.path.abspath(output_dir) + "\n")
            f.write(f"\nFull CSV saved to: {csv_path}\n")
        print(f"Summary written to {summary_path}")
        print(f"CSV written to {csv_path}")
    else:
        print("No results.")

    # --- Compute optimal thresholds per method ---
    if not df.empty:
        print("\nOptimal Thresholds per Method:")
        for method in df["Method"].unique():
            method_df = df[df["Method"] == method]
            scores = method_df["Score"].tolist()
            labels = [1 if s > 0.5 else 0 for s in scores]  # assume 0.5 as initial truth
            best_t, best_acc = find_optimal_threshold(scores, labels)
            print(f"{method}: best threshold={best_t:.3f}, accuracy={best_acc:.3f}")

# --- 4. Checker for Detection Without LALM Manipulation ---
def run_detector_checker(audio_dir: str, watermarks: list[str], filecount: int):
    """
    Runs watermark detectors directly on clean audio (no LALM manipulation)
    to verify that detectors can detect their own watermarks.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running Detector Checker (No LALM Manipulation) on {device} ---")

    wm_classes = {
        "AudioSeal": AudioSealWM,
        "WavMark": WavMarkWM,
        "SilentCipher": SilentCipherWM,
        "SemanticPCA": SemanticPCAWM,
        "SemanticCluster": SemanticClusterWM,
        "SemanticRandom": SemanticWM
    }

    watermarkers = [wm_classes[name](device) for name in watermarks if name in wm_classes]
    files = glob.glob(os.path.join(audio_dir, "*.wav")) + glob.glob(os.path.join(audio_dir, "*.mp3"))
    results = []

    filecount = min(filecount, len(files))
    print(f"Found {len(files)} files. Processing first {filecount}...")

    for filepath in tqdm(files[:filecount]):
        filename = os.path.basename(filepath)
        try:
            wav, sr = torchaudio.load(filepath)
            wav = ensure_mono(wav) 
            if wav.shape[-1] > sr * 5:
                wav = wav[:, :sr * 5]
        except:
            continue

        for wm in watermarkers:
            row = {"File": filename, "Method": wm.name, "Detection": "FAIL", "Score": 0.0}
            try:
                # 1. Embed (This returns audio at 16kHz)
                wm_audio, payload = wm.embed(wav, sr)
                
                # 2. Detect (FIX: Pass 16000, not sr)
                # Since wm_audio is already at 16k, we tell the detector 
                # "This is 16k audio".
                # score = wm.detect(wm_audio, 16000, payload)
                score = wm.detect(wm_audio, wm.wm_sr, payload)
                
                row["Score"] = round(score, 3)
                threshold = 0.5 if wm.name == "AudioSeal" else 0.85
                row["Detection"] = "PASS" if score > threshold else "FAIL"
            except Exception as e:
                print(f"Error: {e}")
                row["Detection"] = "ERROR"
            results.append(row)

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("DETECTOR CHECKER (NO LALM MANIPULATION) RESULTS")
    print("=" * 60)
    if not df.empty:
        print("\nSummary:")
        print(df.groupby(["Method", "Detection"]).size())

        # --- Save detection summary to file ---
        summary_path = os.path.join(audio_dir, "detector_checker_summary.txt")
        csv_path = os.path.join(audio_dir, "detector_checker_results.csv")
        df.to_csv(csv_path, index=False)
        with open(summary_path, "w") as f:
            f.write("Detector Checker Summary (Detection)\n")
            f.write("="*60 + "\n")
            f.write(str(df.groupby(["Method", "Detection"]).size()))
            f.write(f"\n\nFull CSV saved to: {csv_path}\n")
        print(f"Summary written to {summary_path}")
        print(f"CSV written to {csv_path}")
    else:
        print("No results.")

if __name__ == "__main__":
    import argparse

    datasets = ["AIR", "Bach10", "Clotho", "DAPS", "DEMAND", "Freischuetz", "GuitarSet", "jaCappella", "LibriSpeech", "MAESTRO", "PCD"]
    parser = argparse.ArgumentParser(description="Watermark Testing Framework")
    parser.add_argument("--datasets", nargs="+", choices=datasets, default=datasets, help="List of datasets to use")
    parser.add_argument("--watermarks", nargs="+", default=["SemanticCluster", "SemanticRandom", "SemanticPCA"], help="List of watermark methods to test")
    parser.add_argument("--filecount", type=int, default=50, help="Number of files to process")
    parser.add_argument("--mode", type=str, choices=["benchmark", "detector", "both"], default="both", help="Run mode: benchmark, detector, or both")

    args = parser.parse_args()

    base_data_dir = "../../dataset"
    base_output_dir = "../results_denoised"

    global_results = []
    for dataset in args.datasets:
        try:
            audio_dir = os.path.join(base_data_dir, dataset)
            output_dir = os.path.join(base_output_dir, dataset)
            print(f"\n=== Running for dataset: {dataset} ===")
            if args.mode == "benchmark":
                run_benchmark(audio_dir, output_dir, args.watermarks, args.filecount)
            elif args.mode == "detector":
                run_detector_checker(audio_dir, args.watermarks, args.filecount)
            elif args.mode == "both":
                print("\n=== Running DETECTABILITY + SURVIVABILITY combined mode ===")
                run_detector_checker(audio_dir, args.watermarks, args.filecount)
                run_benchmark(audio_dir, output_dir, args.watermarks, args.filecount)
                print("\n=== Computing optimal threshold using raw scores across pre/post watermark and post-attack ===")
                det_csv = os.path.join(audio_dir, "detector_checker_results.csv")
                surv_csv = os.path.join(output_dir, "benchmark_results.csv")
                if os.path.exists(det_csv) and os.path.exists(surv_csv):
                    det_df = pd.read_csv(det_csv)
                    surv_df = pd.read_csv(surv_csv)
                    results = []
                    for method in surv_df["Method"].unique():
                        pre_scores = [0.0] * len(surv_df[surv_df["Method"] == method])
                        pre_labels = [0] * len(pre_scores)
                        det_scores = det_df[det_df["Method"] == method]["Score"].tolist()
                        det_labels = [1] * len(det_scores)
                        surv_scores = surv_df[surv_df["Method"] == method]["Score"].tolist()
                        surv_labels = [1] * len(surv_scores)
                        scores = pre_scores + det_scores + surv_scores
                        labels = pre_labels + det_labels + surv_labels
                        best_t, best_acc = find_optimal_threshold(scores, labels)
                        print(f"{method}: optimal threshold={best_t:.3f}, combined accuracy={best_acc:.3f}")
                        results.append({"Dataset": dataset, "Method": method, "OptimalThreshold": best_t, "Accuracy": best_acc})
                    detect_csv = os.path.join(output_dir, "combined_detectability_results.csv")
                    pd.DataFrame(results).to_csv(detect_csv, index=False)
                    print(f"Combined detectability results written to {detect_csv}")
                    global_results.extend(results)
                else:
                    print(f"Missing CSVs for combined threshold computation in {dataset}.")
        except Exception as e:
            print(f"[Warning] Skipping dataset {dataset} due to error: {e}")
            continue

    if args.mode == "both" and global_results:
        df_global = pd.DataFrame(global_results)
        print("\n=== Summary of Optimal Thresholds per Dataset ===")
        dataset_summary = df_global.groupby("Dataset")[["OptimalThreshold", "Accuracy"]].mean()
        print(dataset_summary)
        global_best_t = df_global["OptimalThreshold"].mean()
        print(f"\nGlobal Best Threshold (mean across datasets): {global_best_t:.3f}")
        summary_csv = os.path.join(base_output_dir, "global_threshold_summary.csv")
        df_global.to_csv(summary_csv, index=False)
        print(f"Global threshold summary written to {summary_csv}")