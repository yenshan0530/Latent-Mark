# Watermark Research

Embedding, detection, and robustness benchmarks for Latent-Mark. All commands below are run from `watermark_research/src/`.

## Setup

```bash
# From the repository root
conda env create -f watermark_research/environment.yml
conda activate aw
```

Alternatives:

```bash
# Exact conda spec
conda create -n aw --file watermark_research/conda-spec.txt
conda activate aw

# pip only (the raw_bench submodule must be checked out)
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r watermark_research/requirements.txt
```

## Datasets

Download the evaluation datasets by following the `raw_bench` submodule's README. Every script expects a root folder with one subfolder per dataset, for example:

```
dataset/
├── AIR/
├── Clotho/
├── DAPS/
├── LibriSpeech/
└── ...
```

The default root is `../../dataset` relative to `src/`. Override it with `--base_dir`. `SilentCipher` needs its checkpoint at `raw_bench/wm_ckpts/silent_cipher/44_1_khz/73999_iteration`, which the submodule provides.

## Watermarking methods (`--watermarks`)

| Method | Paper name | Description |
| --- | --- | --- |
| `SemanticCluster` | Latent-Cluster | Shifts the SNAC latent along the axis between the two k-means centroids of the codebook (24 kHz) |
| `SemanticPCA` | Latent-PCA | Shifts along the first principal component of the codebook |
| `SemanticRandom` | Latent-Random | Shifts along a fixed random unit vector |
| `JointManifold` | Latent-Joint | Jointly optimizes one perturbation across several codec latent spaces (Cross-Codec Optimization) |
| `AudioSeal` | baseline | Neural additive watermark (16 kHz) |
| `WavMark` | baseline | Spread-spectrum bit watermark (16 kHz) |
| `SilentCipher` | baseline | Psychoacoustic watermark (44.1 kHz) |

---

## 1. Single-codec benchmark: `watermark_testing.py`

Embeds each watermark, attacks with a SNAC 24 kHz encode-decode round trip, and detects. This is the Table 1 pipeline.

```bash
python watermark_testing.py --mode both \
  --datasets LibriSpeech DAPS \
  --watermarks SemanticCluster SemanticPCA SemanticRandom AudioSeal WavMark SilentCipher \
  --filecount 120
```

| Argument | Default | Description |
| --- | --- | --- |
| `--mode` | `both` | `detector` (embed and detect, no attack), `benchmark` (embed, attack, detect), or `both` (run detector then benchmark and compute the combined detection threshold) |
| `--datasets` | all 11 | Dataset folder names under `--base_dir` |
| `--watermarks` | `SemanticCluster SemanticRandom SemanticPCA` | Methods to test |
| `--filecount` | `50` | Files per dataset |
| `--base_dir` | `../../dataset` | Dataset root |
| `--out` | `../results_snac` | Output root |

Output:

```
$out/$dataset/
├── qwen_benchmark_results.csv          # per-file scores and PASS/FAIL after the SNAC attack
├── qwen_benchmark_summary.txt
├── combined_detectability_results.csv  # optimal threshold and accuracy per method (--mode both)
└── $method/$file_stem/
    ├── 1_original.wav
    ├── 2_watermarked.wav
    ├── 3_lalm_attacked.wav
    └── analysis_plot.png
$out/global_threshold_summary.csv       # thresholds across all datasets (--mode both)
```

Detector-mode CSVs (`detector_checker_results.csv`) are written next to the input audio.

Summarize into a Table 1 layout:

```bash
python summarize_results.py --results_dir ../results_snac --out watermark_summary_table.csv
```

---

## 2. Cross-codec optimization with a chosen attack: `transferbility_testing.py`

Runs `JointManifold` over a selectable set of codec views and attacks with any of several codecs. Thresholds are estimated per attack from clean and watermarked scores before the attack is applied.

```bash
python transferbility_testing.py --mode both \
  --datasets LibriSpeech \
  --watermarks JointManifold SemanticCluster \
  --joint_codecs snac dac44 funcodec \
  --attack all \
  --filecount 120
```

| Argument | Default | Description |
| --- | --- | --- |
| `--mode` | `both` | `detector`, `benchmark`, or `both` |
| `--datasets` | all 11 | Names under `--base_dir`, or explicit paths (shell globs are expanded) |
| `--watermarks` | `JointManifold SemanticCluster` | Methods to test |
| `--joint_codecs` | `snac encodec24 encodec32` | Codec views for `JointManifold`: any of `snac`, `encodec24`, `encodec32`, `dac44`, `funcodec`, `apcodec` |
| `--attack` | `all` | `snac`, `soundstream`, `encodec24`, `encodec32`, `dac44`, `funcodec`, `apcodec`, or `all` |
| `--filecount` | all files | Files per dataset |
| `--base_dir` | `../../dataset` | Dataset root |
| `--out` | `../results_transferability` | Output root |

`funcodec` and `apcodec` require the `funcodec` and `apcodec` Python packages, which are not part of the default environment. When a package is missing the corresponding view or attack is skipped with a warning.

Output (one pair of CSVs per attack, prefixed with the joint codec set):

```
$out/$dataset/
├── benchmark_results_${joint}_${attack}.csv
├── benchmark_summary_${joint}_${attack}.csv
└── $method_${joint}_${attack}/$file_stem/   (1_original / 2_watermarked / 3_lalm_attacked .wav, analysis_plot.png)
```

Print a pass-rate table by joint set and attack:

```bash
python generate_summary.py --results_dir ../results_transferability
```

---

## 3. Optimization-set sweep: `transferbility_testing_all.py`

Sweeps every optimization set and attack codec in one run. Thresholds for the single-codec methods are calibrated on clean audio to a target false-positive rate; `JointManifold` passes when its score after attack exceeds the clean-audio score after the same attack.

```bash
python transferbility_testing_all.py \
  --datasets LibriSpeech DAPS \
  --watermarks JointManifold SemanticCluster SemanticPCA SemanticRandom \
  --opt_set all --attack all \
  --filecount 120 --out ../results_sweep --save_wavs
```

Optimization sets (`--opt_set`):

| Set | Paper name | Codec views |
| --- | --- | --- |
| `Opt_B1` | C1 | `snac_32`, `dac_16`, `dac_44` |
| `Opt_B2` | C2 | `snac_32`, `encodec_24`, `encodec_32` |
| `Opt_Mix` | F1 | `snac_24`, `dac_24`, `encodec_24` |
| `Opt_A1` | | `snac_24`, `dac_16`, `dac_44` |
| `Opt_A2` | | `snac_24`, `encodec_24`, `encodec_32` |

Attack codecs (`--attack`): `snac_44`, `encodec_48`, `dac_24`, or `all`. The single-codec methods ignore `--opt_set` and always use `snac_24`.

| Argument | Default | Description |
| --- | --- | --- |
| `--datasets` | `LibriSpeech Bach10` | Names or paths |
| `--skip_datasets` | none | Names to skip |
| `--base_dir` | `../../dataset` | Dataset root |
| `--out` | `../results_exp15` | Output root |
| `--watermarks` | `JointManifold SemanticCluster` | Methods |
| `--filecount` | `120` | Files per experiment |
| `--calib_files` | `42` | Clean files used for calibration |
| `--fpr` | `0.01` | Target false-positive rate for the single-codec thresholds |
| `--seed` | `0` | Shuffle seed |
| `--save_wavs` | off | Save original / watermarked / attacked triplets |

Output:

```
$out/$dataset/
├── summary_exp15.csv                      # one row per (method, opt set, attack)
├── summary_exp15.txt
├── results_${opt}_${method}_vs_${attack}.csv
└── ${opt}_${method}_vs_${attack}/$file_stem/   (with --save_wavs)
```

---

## 4. DSP attacks: `watermark_against_attacks.py`

Robustness to Gaussian noise, amplitude scaling, low-pass filtering, and resampling (Table 3).

```bash
python watermark_against_attacks.py --mode benchmark \
  --datasets LibriSpeech AIR \
  --watermarks all --filecount 120
```

| Argument | Default | Description |
| --- | --- | --- |
| `--mode` | `benchmark` | `benchmark`, `detector`, or `both` |
| `--datasets` | all 11 | Names under `--base_dir` |
| `--watermarks` | `SemanticCluster SemanticRandom SemanticPCA` | Methods, or `all` |
| `--filecount` | `1000` | Files per dataset |
| `--base_dir` | `../../dataset` | Dataset root |
| `--out` | `../results_dsp` | Output root |

Results are written to `$out/$dataset/general_attack_results.csv` with one row per file, method, and attack.

---

## 5. Audio quality

See `audio_quality_check/README.md`. In short:

```bash
cd ../../audio_quality_check
python evaluate_quality.py --dir ../watermark_research/results_snac --out quality_results.csv
python plot_for_paper.py --delta_si_snr_csv quality_results.csv --utmos_csv quality_results.csv --out plots
```
