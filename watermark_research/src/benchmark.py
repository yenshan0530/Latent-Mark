#!/usr/bin/env python
"""
Detectability, survivability and transferability benchmark for Latent-Mark (Section 4 of the paper).

For every dataset and every method:
  1. calibrate the detector on clean audio (Latent-Mark methods only)
  2. for each test file: score the clean file (NEG) and the watermarked file (POS)   -> Detectability
  3. for each attack: score the attacked watermarked file                            -> Survivability
     (the attacked clean file is scored too, so the Delta-Score of Section 3.3 is available)

Examples
  # Table 1: SNAC 24 kHz compression, all methods
  python benchmark.py --datasets LibriSpeech DAPS --methods Latent-Cluster Latent-PCA Latent-Random Latent-Joint \
      AudioSeal WavMark SilentCipher --joint_views snac_32 dac_16 dac_44 --attacks snac_24 --filecount 120

  # Table 2: cross-codec optimization set C1, unseen codecs
  python benchmark.py --datasets Clotho --methods Latent-Joint --joint_views snac_32 dac_16 dac_44 \
      --attacks snac_44 encodec_48 dac_24 --filecount 120 --tag C1

  # Table 3: DSP attacks
  python benchmark.py --datasets AIR --methods Latent-Cluster AudioSeal WavMark SilentCipher \
      --attacks gaussian amplitude lowpass resample --filecount 120
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import latentmark as lm  # noqa: E402

torch.backends.cudnn.benchmark = False

DEFAULT_DATASETS = ["AIR", "Clotho", "DAPS", "LibriSpeech", "PCD", "jaCappella", "MAESTRO", "GuitarSet", "Freischuetz"]


def build_method(name: str, args, device: str) -> lm.Watermarker:
    name = lm.canonical_method(name)
    common = dict(k=args.k, gamma=args.gamma, steps=args.steps, lr=args.lr, beta=args.beta, target_sdr=args.sdr,
                  eps_min=args.eps_min, eps_max=args.eps_max, pad_multiple=args.pad_multiple,
                  frames_per_file=args.calib_frames, seed=args.seed, target_mode=args.target_mode,
                  sigma_level=args.sigma_level, hinge=args.hinge)
    if name == "Latent-Cluster":
        return lm.LatentMark([args.single_view], device, axis="cluster", **common)
    if name == "Latent-PCA":
        return lm.LatentMark([args.single_view], device, axis="pca", **common)
    if name == "Latent-Random":
        return lm.LatentMark([args.single_view], device, axis="random", **common)
    if name == "Latent-Joint":
        return lm.LatentMark(args.joint_views, device, axis=args.joint_axis, work_sr=args.work_sr, **common)
    if name == "AudioSeal":
        return lm.AudioSealWM(device)
    if name == "WavMark":
        return lm.WavMarkWM(device, seed=args.seed)
    if name == "SilentCipher":
        return lm.SilentCipherWM(device, ckpt_dir=args.silentcipher_ckpt)
    raise ValueError(name)


def resolve_dataset(name: str, base_dir: str):
    if os.path.isdir(name):
        return os.path.basename(os.path.normpath(name)), name
    path = os.path.join(base_dir, name)
    if os.path.isdir(path):
        return name, path
    return name, None


def split_files(files, filecount, calib_files, seed):
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)
    test = files[:filecount]
    rest = files[filecount:]
    if len(rest) >= calib_files:
        calib = rest[:calib_files]
        overlap = False
    else:
        calib = (rest + test)[:calib_files]
        overlap = True
    return test, calib, overlap


def summarize(df: pd.DataFrame, attacks, condition: str = None) -> pd.DataFrame:
    """Per method: detectability on clean vs watermarked (Det. acc, TPR, FPR), survivability per attack
    (score after attack above the detection threshold), and transferability per attack (Delta-Score > 0, Section 3.3:
    score(R_a(s_wm)) - score(R_a(s)) > 0). With `condition`, transferability is computed only over files whose
    watermark survived that attack, as in Section 5.2."""
    rows = []
    for method, g in df.groupby("method", sort=False):
        g = g[g["error"].isna()] if "error" in g else g
        n = len(g)
        if n == 0:
            continue
        thr = float(g["threshold"].iloc[0])
        neg_pass = (g["clean_score"] > thr)
        pos_pass = (g["wm_score"] > thr)
        row = {"method": method, "n": n,
               "det_acc": float((pos_pass.sum() + (~neg_pass).sum()) / (2 * n)),
               "tpr": float(pos_pass.mean()), "fpr": float(neg_pass.mean())}
        cond_mask = None
        if condition and f"attacked_score__{condition}" in g:
            cond_mask = g[f"attacked_score__{condition}"] > thr
            row["n_cond"] = int(cond_mask.sum())
        for a in attacks:
            col, dcol = f"attacked_score__{a}", f"clean_attacked_score__{a}"
            if col not in g:
                continue
            row[f"sur__{a}"] = float((g[col] > thr).mean())
            if dcol in g:
                delta = g[col] - g[dcol]
                sel = delta[cond_mask] if cond_mask is not None else delta
                row[f"xfer__{a}"] = float((sel > 0).mean()) if len(sel) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="dataset names under --base_dir, or paths")
    ap.add_argument("--base_dir", default="../../dataset")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. an optimization-set name")
    ap.add_argument("--methods", nargs="+",
                    default=["Latent-Cluster", "Latent-PCA", "Latent-Random", "Latent-Joint", "AudioSeal", "WavMark", "SilentCipher"])
    ap.add_argument("--attacks", nargs="+", default=["snac_24"],
                    help=f"codec views {sorted(lm.VIEW_SPECS)} and/or DSP attacks {list(lm.DSP_ATTACKS)}")
    ap.add_argument("--filecount", type=int, default=120, help="test files per dataset (each gives one NEG and one POS sample)")
    ap.add_argument("--xfer_condition", default=None, metavar="ATTACK",
                    help="report transferability only over files whose watermark survived this attack (e.g. snac_24), as in Section 5.2")
    ap.add_argument("--calib_files", type=int, default=42, help="clean files for the null distribution, disjoint from the test files when possible")
    ap.add_argument("--seconds", type=float, default=5.0, help="crop every file to its first N seconds (0 = full length)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_wavs", action="store_true", help="write 1_original / 2_watermarked / 3_attacked wavs per file")
    ap.add_argument("--plots", action="store_true", help="with --save_wavs, also write analysis_plot.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Latent-Mark hyperparameters (Section 3 defaults)
    g = ap.add_argument_group("Latent-Mark")
    g.add_argument("--single_view", default="snac_24", help="codec for Latent-Cluster / PCA / Random")
    g.add_argument("--joint_views", nargs="+", default=["snac_32", "dac_16", "dac_44"], help="surrogate committee for Latent-Joint (paper set C1)")
    g.add_argument("--joint_axis", default="cluster", choices=["cluster", "pca", "random"])
    g.add_argument("--work_sr", type=int, default=44100, help="f_work for Latent-Joint")
    g.add_argument("--k", type=float, default=1.5, help="tau = mu + k sigma")
    g.add_argument("--gamma", type=float, default=1.5, help="target alignment score of Eq. 5 (single codec)")
    g.add_argument("--target_mode", default="margin", choices=["margin", "gamma", "tau"],
                   help="embedding target per view: 'margin' = tau_c + gamma*sigma_c (default), 'tau' = tau_c with no safety margin, "
                        "'gamma' = the absolute projection value gamma")
    g.add_argument("--sigma_level", default="file", choices=["file", "frame"],
                   help="null-distribution std: over clean-clip means of the projection (default) or over individual frames")
    g.add_argument("--hinge", default="mean", choices=["mean", "frame"], help="hinge on the clip-mean projection (default) or per frame")
    g.add_argument("--steps", type=int, default=150)
    g.add_argument("--lr", type=float, default=5e-3)
    g.add_argument("--beta", type=float, default=2.5)
    g.add_argument("--sdr", type=float, default=42.0, help="target SDR (dB) that sets the L-inf budget epsilon")
    g.add_argument("--eps_min", type=float, default=1e-4)
    g.add_argument("--eps_max", type=float, default=0.1)
    g.add_argument("--pad_multiple", type=int, default=4096)
    g.add_argument("--calib_frames", type=int, default=512, help="frame projections sampled per calibration file")
    # DSP attack strengths (Table 3)
    d = ap.add_argument_group("DSP attacks")
    d.add_argument("--snr_db", type=float, default=60.0)
    d.add_argument("--amp_ratio", type=float, default=0.5)
    d.add_argument("--lowpass_hz", type=float, default=4000.0)
    d.add_argument("--resample_hz", type=int, default=16000)
    ap.add_argument("--silentcipher_ckpt", default=None, help="folder with the 44.1 kHz SilentCipher checkpoint (default: raw_bench submodule, else Hugging Face)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seconds = args.seconds if args.seconds > 0 else None
    out_root = args.out + (f"_{args.tag}" if args.tag else "")
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    attacks = {a: lm.make_attack(a, args.device, args.snr_db, args.amp_ratio, args.lowpass_hz, args.resample_hz) for a in args.attacks}
    methods = {}
    for m in args.methods:
        try:
            methods[lm.canonical_method(m)] = build_method(m, args, args.device)
        except Exception as e:
            print(f"[skip] {m}: {e}")

    all_summaries = []
    for ds in args.datasets:
        name, path = resolve_dataset(ds, args.base_dir)
        if path is None:
            print(f"[skip] dataset not found: {ds}")
            continue
        files = lm.list_audio_files(path)
        if not files:
            print(f"[skip] no audio in {path}")
            continue
        test, calib, overlap = split_files(files, args.filecount, args.calib_files, args.seed)
        print(f"\n=== {name}: {len(test)} test files, {len(calib)} calibration files"
              + (" (calibration overlaps the test set: dataset is small)" if overlap else "") + " ===")
        ds_out = os.path.join(out_root, name)
        os.makedirs(ds_out, exist_ok=True)

        for wm in methods.values():
            t0 = time.time()
            wm.calibrate(calib, seconds or 5.0)
            if time.time() - t0 > 1:
                print(f"[calib] {wm.name}: {time.time() - t0:.0f}s")

        rows = []
        for path_i in tqdm(test, desc=name):
            stem = os.path.splitext(os.path.basename(path_i))[0]
            try:
                wav, sr = lm.load_audio(path_i, seconds)
            except Exception as e:
                print(f"[skip] {path_i}: {e}")
                continue
            for wm in methods.values():
                row = {"dataset": name, "file": os.path.basename(path_i), "method": wm.name, "threshold": wm.threshold}
                try:
                    t0 = time.time()
                    row["clean_score"] = wm.detect(wav, sr, None)
                    wm_wav, payload = wm.embed(wav, sr)
                    row["embed_s"] = round(time.time() - t0, 2)
                    row["wm_score"] = wm.detect(wm_wav, wm.wm_sr, payload)
                    clean_at_wm_sr = lm.resample(wav, sr, wm.wm_sr)
                    attacked_first = None
                    for aname, atk in attacks.items():
                        attacked = atk(wm_wav, wm.wm_sr)
                        row[f"attacked_score__{aname}"] = wm.detect(attacked, wm.wm_sr, payload)
                        clean_attacked = atk(clean_at_wm_sr, wm.wm_sr)
                        row[f"clean_attacked_score__{aname}"] = wm.detect(clean_attacked, wm.wm_sr, payload)
                        if args.save_wavs:
                            folder = os.path.join(ds_out, wm.name, stem)
                            os.makedirs(folder, exist_ok=True)
                            torchaudio.save(os.path.join(folder, f"3_attacked_{aname}.wav"), lm.as_bct(attacked)[0].cpu(), wm.wm_sr)
                        if attacked_first is None:
                            attacked_first = attacked
                    if args.save_wavs:
                        lm.save_artifacts(out_root, name, wm.name, stem, wav, sr, wm_wav, wm.wm_sr, attacked_first, plot=args.plots)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                    traceback.print_exc()
                rows.append(row)

        df = pd.DataFrame(rows)
        if "error" not in df:
            df["error"] = np.nan
        df.to_csv(os.path.join(ds_out, "scores.csv"), index=False)
        summ = summarize(df, list(attacks), condition=args.xfer_condition)
        summ.insert(0, "dataset", name)
        summ.to_csv(os.path.join(ds_out, "summary.csv"), index=False)
        all_summaries.append(summ)
        n_err = int(df["error"].notna().sum())
        print(f"\n--- {name} ---" + (f"  ({n_err} errors, see scores.csv)" if n_err else ""))
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if all_summaries:
        allsum = pd.concat(all_summaries, ignore_index=True)
        allsum.to_csv(os.path.join(out_root, "summary_all.csv"), index=False)
        print(f"\nWrote {os.path.join(out_root, 'summary_all.csv')}")


if __name__ == "__main__":
    main()
