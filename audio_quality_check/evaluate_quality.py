import os
import glob
import numpy as np
import soundfile as sf
import librosa
from scipy import stats
import math
import csv

# Optional libs (pip install pesq pystoi mir_eval tqdm)
try:
    from pesq import pesq
except ImportError:
    pesq = None
try:
    from pystoi import stoi
except ImportError:
    stoi = None
try:
    from tqdm import tqdm  # Import progress bar package
except ImportError:
    tqdm = None
try:
    import torch
except ImportError:
    torch = None


def load_audio(path, sr=16000, mono=True):
    y, fs = sf.read(path)
    if mono and y.ndim == 2:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    if fs != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=fs, target_sr=sr)
        fs = sr
    return y, fs


def si_snr(est, ref, eps=1e-8):
    assert est.shape == ref.shape
    ref = ref.astype(np.float32)
    est = est.astype(np.float32)
    s_target = np.sum(ref * est) * ref / (np.sum(ref**2) + eps)
    e_noise = est - s_target
    return 10 * np.log10((np.sum(s_target**2) + eps) / (np.sum(e_noise**2) + eps))


def simple_snr(ref, est, eps=1e-8):
    sig = np.sum(ref**2)
    noise = np.sum((ref - est) ** 2)
    return 10 * np.log10((sig + eps) / (noise + eps))


def log_spectral_distance(ref, est, n_fft=1024, hop=512, eps=1e-8):
    ref_spec = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop)) + eps
    est_spec = np.abs(librosa.stft(est, n_fft=n_fft, hop_length=hop)) + eps
    ref_db = 20 * np.log10(ref_spec)
    est_db = 20 * np.log10(est_spec)
    lsd = np.sqrt(np.mean((ref_db - est_db) ** 2, axis=0))  # per frame
    return float(np.mean(lsd))


def compute_metrics(clean_path, wm_path, sr=16000, utmos_predictor=None, device=None):
    clean, fs = load_audio(clean_path, sr=sr)
    wm, _ = load_audio(wm_path, sr=sr)
    # align lengths
    L = min(len(clean), len(wm))
    clean = clean[:L]
    wm = wm[:L]
    metrics = {}
    metrics["si_snr_clean"] = si_snr(
        clean, clean
    )  # trivially inf/large, keep as baseline
    metrics["si_snr_watermarked"] = si_snr(wm, clean)
    metrics["delta_si_snr"] = metrics["si_snr_watermarked"] - metrics["si_snr_clean"]
    metrics["snr"] = simple_snr(clean, wm)
    metrics["lsd"] = log_spectral_distance(clean, wm)

    if pesq is not None:
        try:
            # pesq requires fs 16000 or 8000 for narrowband, some versions support wideband
            metrics["pesq"] = pesq(fs, clean, wm, "wb")
        except Exception:
            metrics["pesq"] = None

    if stoi is not None:
        try:
            metrics["stoi"] = stoi(clean, wm, fs, extended=False)
        except Exception:
            metrics["stoi"] = None

    # Calculate UTMOS
    metrics["utmos_clean"] = None
    metrics["utmos_watermarked"] = None
    if utmos_predictor is not None and torch is not None:
        try:
            # Check for NaN or Inf
            if np.isnan(wm).any() or np.isinf(wm).any():
                print(f"Warning: Watermarked audio contains NaN or Inf: {wm_path}")
            else:
                clean_tensor = torch.from_numpy(clean).float().unsqueeze(0).to(device)
                wm_tensor = torch.from_numpy(wm).float().unsqueeze(0).to(device)

            with torch.no_grad():
                metrics["utmos_clean"] = utmos_predictor(clean_tensor, fs).item()
                metrics["utmos_watermarked"] = utmos_predictor(wm_tensor, fs).item()
                metrics["delta_utmos"] = (
                    metrics["utmos_watermarked"] - metrics["utmos_clean"]
                )

        except Exception as e:
            print(f"UTMOS Error for {wm_path}: {e}")
            metrics["delta_utmos"] = None

    return metrics


def evaluate_pair_list(
    pairs, out_csv="quality_results.csv", sr=16000, utmos_predictor=None, device=None
):
    results = []
    total_pairs = len(pairs)

    # Determine whether to use the tqdm progress bar
    iterator = (
        tqdm(pairs, desc="Processing Audio Pairs", unit="file") if tqdm else pairs
    )

    for i, row in enumerate(iterator):
        clean_path = row["clean"]
        wm_path = row["watermarked"]

        # If there is no tqdm, print the traditional [1/100] progress text
        if not tqdm:
            print(f"[{i+1}/{total_pairs}] Evaluating: {clean_path} vs {wm_path}")

        if not os.path.exists(clean_path):
            if not tqdm:
                print("CLEAN NOT FOUND:", clean_path)
            continue
        if not os.path.exists(wm_path):
            if not tqdm:
                print("WM NOT FOUND:", wm_path)
            continue

        wm_method = row.get("wm_method", "")
        instrument = row.get("instrument", "")

        try:
            m = compute_metrics(
                clean_path,
                wm_path,
                sr=sr,
                utmos_predictor=utmos_predictor,
                device=device,
            )
        except Exception as e:
            if not tqdm:
                print("METRIC ERROR for:", clean_path, wm_path, "Error:", e)
            continue

        out = {
            "clean": clean_path,
            "watermarked": wm_path,
            "wm_method": wm_method,
            "instrument": instrument,
        }
        out.update(m)
        results.append(out)

    if not results:
        print("No valid results computed.")
        return []

    # Save file
    keys = list(results[0].keys())
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return results


def get_pairs_from_csv(csv_path):
    pairs = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pairs.append(row)
    return pairs


def get_pairs_from_dir(dir_path):
    pairs = []
    for root, dirs, files in os.walk(dir_path):
        if "1_original.wav" in files and "2_watermarked.wav" in files:
            clean_path = os.path.join(root, "1_original.wav")
            wm_path = os.path.join(root, "2_watermarked.wav")
            folder_name = os.path.basename(os.path.dirname(root))
            if "Semantic" in folder_name:
                continue
            pairs.append(
                {
                    "clean": clean_path,
                    "watermarked": wm_path,
                    "wm_method": folder_name,
                    "instrument": "unknown",
                }
            )
    return pairs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Audio Quality Evaluator")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--csv", help="CSV with columns: clean,watermarked,wm_method,instrument"
    )
    group.add_argument(
        "--dir",
        help="Directory to scan recursively for 1_original.wav and 2_watermarked.wav",
    )
    p.add_argument("--out", default="quality_results.csv", help="Output CSV path")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = p.parse_args()

    # --------------------------------
    # Initialize the UTMOS model (load only once)
    # --------------------------------
    utmos_predictor = None
    device = None
    if torch is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading UTMOS model to {device}...")
        try:
            utmos_predictor = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            ).to(device)
            utmos_predictor.eval()  # Set to evaluation mode
            print("✅ UTMOS model loaded successfully!")
        except Exception as e:
            print(f"Failed to load UTMOS model, skipping this metric. Error: {e}")
    else:
        print("PyTorch is not installed, skipping UTMOS calculation.")
    print("-" * 40)

    pairs = []
    if args.csv:
        print(f"Reading pairs from CSV: {args.csv}")
        pairs = get_pairs_from_csv(args.csv)
    elif args.dir:
        print(f"Scanning directory for pairs: {args.dir}")
        pairs = get_pairs_from_dir(args.dir)

    print(f"Found {len(pairs)} pairs to evaluate.\n")

    if pairs:
        res = evaluate_pair_list(
            pairs,
            out_csv=args.out,
            sr=args.sr,
            utmos_predictor=utmos_predictor,
            device=device,
        )
        print(f"\nDone! Wrote {len(res)} valid results to {args.out}")
    else:
        print("Nothing to evaluate.")