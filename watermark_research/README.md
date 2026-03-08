
# Watermark Research

### Setup
```
conda create -n audio-watermark python=3.12
conda activate audio-watermark
pip install -r requirements.txt
```

```
# Transferbility
# Default
conda env create -f environment.yml
conda activate aw

# conda
conda create -n aw --file conda-spec.txt
conda activate aw

# pip only
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Datasets
Download and prepare the datasets as described in `raw_bench`. Ensure that the datasets are organized under a common root directory (named as `../../test_dats/`) with subfolders for each dataset.


### Watermark Testing (Single-Attacker Benchmark)

```bash
cd watermark_research/src
python watermark_testing.py \
  --mode benchmark \
  --datasets LibriSpeech Bach10 \
  --watermarks SemanticCluster SemanticRandom SemanticPCA \
  --filecount 120
```

#### What This Script Does

For each dataset × watermark method combination, the script runs in three modes:

- **`benchmark`**: Embeds a watermark, applies a SNAC codec roundtrip attack, then detects whether the watermark survived. Saves audio triplets and analysis plots.
- **`detector`**: Embeds a watermark and detects it directly (no attack) to verify each method's baseline detectability.
- **`both`**: Runs detector and benchmark sequentially, then computes the optimal detection threshold by combining pre-watermark, post-watermark, and post-attack scores.

#### Watermarking Methods (`--watermarks`)

| Method | Description |
|---|---|
| `AudioSeal` | Neural watermarking via additive perturbation (16 kHz) |
| `WavMark` | Bit-level watermarking via spread-spectrum encoding (16 kHz) |
| `SilentCipher` | Psychoacoustic watermarking (44.1 kHz); requires checkpoint at `raw_bench/wm_ckpts/silent_cipher/` |
| `SemanticPCA` | Steers SNAC latent toward the top PCA direction of the codebook (24 kHz) |
| `SemanticCluster` | Steers SNAC latent toward the K-Means inter-centroid axis (24 kHz) |
| `SemanticRandom` | Steers SNAC latent toward a fixed random unit vector (24 kHz) |

#### Attack

The sole attacker is a full SNAC 24 kHz encode→decode roundtrip simulating the tokenization pipeline.

#### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `both` | `benchmark`, `detector`, or `both` |
| `--datasets` | all 11 datasets | Dataset folder names under `../../dataset/` |
| `--watermarks` | `SemanticCluster SemanticRandom SemanticPCA` | Watermarking methods to test |
| `--filecount` | `50` | Number of audio files to process per dataset |

#### Output Structure

```
../results_denoised/
└── $dataset_name/
    ├── benchmark_summary.txt      # Survivability counts per method
    ├── benchmark_results.csv      # Per-file scores and PASS/FAIL
    ├── combined_detectability_results.csv  # Optimal thresholds (--mode both)
    └── $method_name/
        └── $file_stem/
            ├── 1_original.wav
            ├── 2_watermarked.wav
            ├── 3_lalm_attacked.wav
            └── analysis_plot.png       # Waveform, spectrogram, and residual plots
```

For `--mode detector`, results are saved alongside the input audio:

```
../../dataset/$dataset_name/
├── detector_checker_summary.txt
└── detector_checker_results.csv
```

#### Example

```bash
# Run full benchmark (attack + detect) on two datasets
python watermark_testing.py --mode benchmark --datasets LibriSpeech Bach10 --watermarks SemanticCluster SemanticRandom --filecount 120

# Check baseline detectability only (no attack)
python watermark_testing.py --mode detector --datasets LibriSpeech --watermarks SemanticCluster SemanticRandom --filecount 120

# Run both and compute optimal detection thresholds
python watermark_testing.py --mode both --datasets LibriSpeech Bach10 --watermarks SemanticCluster SemanticRandom SemanticPCA --filecount 120
```

---

### Joint Optimization (Transferability)

```bash
cd watermark_research/src
python watermark_testing_joint.py \
  --mode both \
  --datasets LibriSpeech Bach10 \
  --watermarks JointManifold SemanticCluster \
  --filecount 50 \
  --attack snac
```

#### What This Script Does

For each dataset × watermark method combination, the script supports three modes:

- **`detector`**: Embeds a watermark and scores both clean (NEG) and watermarked (POS) audio to estimate optimal detection thresholds. No attack applied.
- **`benchmark`**: First runs the detector phase to estimate thresholds, then applies a codec attack and checks whether the watermark survives.
- **`both`**: Runs detector and benchmark sequentially, then merges NEG, pre-attack POS, and post-attack POS scores to compute a combined optimal threshold per method.

The key addition over `watermark_testing.py` is the **`JointManifoldWM`** method and a configurable **`AttackRouter`** supporting multiple codec backends.

#### Watermarking Methods (`--watermarks`)

| Method | Description |
|---|---|
| `JointManifold` | Jointly optimizes a perturbation across multiple codec latent spaces (EnCodec 24k/32k, DAC 44k, SNAC). Requires `calibrate_from_audio_dir()` before use. |
| `SemanticPCA` | Steers SNAC latent toward the top PCA direction of the codebook (24 kHz) |
| `SemanticCluster` | Steers SNAC latent toward the K-Means inter-centroid axis (24 kHz) |
| `SemanticRandom` | Steers SNAC latent toward a fixed random unit vector (24 kHz) |
| `AudioSeal` | Neural watermarking via additive perturbation (16 kHz) |
| `WavMark` | Bit-level spread-spectrum watermarking (16 kHz) |
| `SilentCipher` | Psychoacoustic watermarking (44.1 kHz); requires checkpoint at `raw_bench/wm_ckpts/silent_cipher/` |

#### Attack Codecs (`--attack`)

| Value | Description |
|---|---|
| `snac` | SNAC 24 kHz encode→decode (default) |
| `encodec24` | EnCodec 24 kHz encode→decode |
| `encodec32` | EnCodec 32 kHz encode→decode |
| `dac44` | DAC 44.1 kHz encode→decode |
| `soundstream` | SoundStream 16 kHz encode→decode |

#### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `both` | `benchmark`, `detector`, or `both` |
| `--datasets` | all 11 datasets | Dataset folder names or full paths (supports glob) |
| `--watermarks` | `JointManifold SemanticCluster` | Watermarking methods to test |
| `--filecount` | `None` (all files) | Number of audio files to process per dataset |
| `--attack` | `snac` | Codec attack backend to use |

#### Output Structure

```
$base_output_dir/
├── global_threshold_summary.csv         # Aggregated optimal thresholds across all datasets
└── $dataset_name/
    ├── benchmark_results.csv        # Per-file scores and PASS/FAIL (benchmark/both)
    ├── combined_detectability_results.csv  # Combined NEG+POS+attacked thresholds (both)
    └── $method_name/
        └── $file_stem/
            ├── 1_original.wav
            ├── 2_watermarked.wav
            ├── 3_lalm_attacked.wav
            └── analysis_plot.png         # Waveform, spectrogram, and residual plots
```

For `--mode detector` and `--mode both`, threshold analysis files are saved alongside the input audio:

```
$audio_dir/
├── detector_checker_results.csv
└── detector_checker_thresholds_clean_$attack.csv
```

#### Example

```bash
# Run full pipeline: threshold estimation + attack survivability
python watermark_testing_joint.py --mode both --datasets LibriSpeech Bach10 --watermarks JointManifold SemanticCluster --filecount 50 --attack snac

# Estimate detection thresholds only (no attack)
python watermark_testing_joint.py --mode detector --datasets LibriSpeech --watermarks JointManifold --filecount 30 --attack encodec24

# Run benchmark with custom dataset path
python watermark_testing_joint.py --mode benchmark --datasets ../../raw_bench/test_data/GuitarSet --watermarks SemanticCluster SemanticRandom --attack dac44
```



---

### Transferability for All Methods and Datasets in Single Run

```bash
cd watermark_research/src
python transferbility_testing_all.py \
  --datasets $DATASET_NAMES \
  --base_dir $DATASET_ROOT \
  --watermarks JointManifold SemanticPCA SemanticCluster SemanticRandom \
  --filecount 120 \
  --opt_set all \
  --attack all \
  --out $OUTPUT_ROOT \
  --skip_datasets $SKIPPED_DATASETS \
  --save_wavs
```

#### What This Script Does

For each dataset × watermark method × optimization set × attack codec combination, the script:

1. **Calibrates** a detection threshold on clean audio (controls empirical FPR ≤ `--fpr`)
2. **Embeds** a watermark into each audio file
3. **Attacks** the watermarked audio via codec roundtrip (encode → decode)
4. **Detects** whether the watermark survives the attack
5. **Saves** per-experiment CSVs and an overall `summary_exp15.csv` per dataset

#### Watermarking Methods (`--watermarks`)

| Method | Description |
|---|---|
| `JointManifold` | Optimizes perturbation jointly across multiple codec latent spaces |
| `SemanticPCA` | Steers the SNAC latent toward the top PCA direction of the codebook |
| `SemanticCluster` | Steers toward the K-Means inter-centroid axis of the SNAC codebook |
| `SemanticRandom` | Steers toward a fixed random axis in the SNAC latent space |

#### Optimization Sets (`--opt_set`)

Each set defines which codec latent spaces are jointly optimized during `JointManifold` embedding:

| Set | Codecs |
|---|---|
| `Opt_A1` | `snac_24`, `dac_16`, `dac_44` |
| `Opt_A2` | `snac_24`, `encodec_24`, `encodec_32` |
| `Opt_Mix` | `snac_24`, `dac_24`, `encodec_24` |
| `Opt_B1` | `snac_32`, `dac_16`, `dac_44` |
| `Opt_B2` | `snac_32`, `encodec_24`, `encodec_32` |

Use `--opt_set all` to run all five sets. For Semantic methods, the optimization set argument is accepted but has no effect (they only use `snac_24`).

#### Attack Codecs (`--attack`)

Each attack applies a full encode–decode roundtrip through the specified codec as a removal attempt:

| Attack | Codec |
|---|---|
| `snac_44` | SNAC 44 kHz |
| `encodec_48` | EnCodec 48 kHz |
| `dac_24` | DAC 24 kHz |

Use `--attack all` to run all three attacks.

#### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `LibriSpeech Bach10` | Dataset folder names or paths to evaluate |
| `--base_dir` | `../../dataset` | Root directory for resolving dataset names |
| `--out` | `../results_exp15` | Root directory for all output CSVs and audio artifacts |
| `--watermarks` | `JointManifold SemanticCluster` | Watermarking methods to test |
| `--filecount` | `120` | Number of audio files to process per experiment |
| `--calib_files` | `42` | Number of clean files used for threshold calibration |
| `--opt_set` | `all` | Which optimization set(s) to run (`all` or a single key) |
| `--attack` | `all` | Which attack codec(s) to run (`all` or a single codec name) |
| `--fpr` | `0.01` | Target false positive rate for threshold calibration |
| `--seed` | `0` | Random seed for file shuffling and reproducibility |
| `--save_wavs` | `False` | If set, saves `original / watermarked / attacked` WAV triplets |
| `--skip_datasets` | `[]` | Dataset names to skip (matched against folder basename) |

#### Output Structure

```
$out/
└── $dataset_name/
    ├── summary_exp15.csv          # Aggregated stats across all experiments
    ├── summary_exp15.txt          # Human-readable version of the summary table
    └── results_$opt_$wm_vs_$atk.csv  # Per-file scores and pass/fail for each experiment
```

If `--save_wavs` is set, audio artifacts are saved alongside:

```
$out/
└── $dataset_name/
    └── $opt_$wm_vs_$atk/
        └── $file_stem/
            ├── 1_original.wav
            ├── 2_watermarked.wav
            └── 3_attacked.wav
```

#### Example

```bash
python transferbility_testing_all.py \
  --datasets LibriSpeech Bach10 \
  --base_dir ../../raw_bench/test_data \
  --watermarks JointManifold SemanticPCA SemanticCluster SemanticRandom \
  --filecount 120 \
  --opt_set all \
  --attack all \
  --out ../../results_exp15 \
  --skip_datasets AIR Clotho DAPS DEMAND Freischuetz GuitarSet jaCappella MAESTRO PCD \
  --save_wavs
```

