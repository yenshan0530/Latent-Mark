# Watermark research

Embedding, detection and robustness benchmarks for Latent-Mark. Two files carry the whole pipeline:

| File | Role |
| --- | --- |
| `src/latentmark.py` | The method (Section 3 of the paper): codec views, secret axes, calibration, `LatentMark` embed/detect, baseline wrappers, codec and DSP attacks |
| `src/benchmark.py` | The experiments (Section 4 and 5): detectability, survivability and transferability for any set of methods, attacks and datasets |
| `src/summarize.py` | Turns one or more result folders into paper-style tables |
| `legacy/` | The original per-experiment scripts, kept for reference |

All commands below are run from `watermark_research/src/` inside the `latentmark` environment (`bash setup_env.sh` at the repository root creates it).

## Datasets

Every script expects a root folder with one subfolder per dataset. The default root is `../../dataset`; override it with `--base_dir`, or pass dataset paths directly.

```
dataset/
├── AIR/  ├── Clotho/  ├── DAPS/  ├── LibriSpeech/  ├── PCD/  ├── jaCappella/  ├── MAESTRO/  ├── GuitarSet/  └── Freischuetz/
```

The evaluation sets come from RAW-Bench (`raw_bench` submodule; follow its README to download them). Codec weights (SNAC, DAC, EnCodec) and the AudioSeal, WavMark and SilentCipher checkpoints download automatically on first use.

## Methods (`--methods`)

| Name | Description |
| --- | --- |
| `Latent-Cluster` | Shift along the axis between the two k-means centroids of the codec's first codebook (Eq. 7). The main method. |
| `Latent-PCA` | Shift along the first principal component of the codebook |
| `Latent-Random` | Shift along a fixed random unit vector |
| `Latent-Joint` | Cross-Codec Optimization: one perturbation optimized jointly across several surrogate codecs (`--joint_views`) |
| `AudioSeal`, `WavMark`, `SilentCipher` | Baselines, detected with each method's own rule |

The single-codec methods use `--single_view` (default `snac_24`). Codec views: `snac_24`, `snac_32`, `snac_44`, `dac_16`, `dac_24`, `dac_44`, `encodec_24`, `encodec_32`, `encodec_48`.

## Attacks (`--attacks`)

Any codec view name runs a full encode-quantize-decode round trip through that codec. The DSP attacks of Table 3 are `gaussian` (`--snr_db`, default 60), `amplitude` (`--amp_ratio`, 0.5), `lowpass` (`--lowpass_hz`, 4000) and `resample` (`--resample_hz`, 16000).

## Running the benchmark

```bash
# Table 1: detectability and survivability under SNAC 24 kHz compression
python benchmark.py --datasets LibriSpeech DAPS AIR Clotho PCD jaCappella MAESTRO GuitarSet \
    --methods Latent-Cluster Latent-PCA Latent-Random Latent-Joint AudioSeal WavMark SilentCipher \
    --joint_views snac_32 dac_16 dac_44 --attacks snac_24 --filecount 120 --out ../results_table1

# Table 2: one optimization set, attacked with unseen codecs (repeat with --tag C2 / F1 and other --joint_views)
python benchmark.py --datasets Clotho LibriSpeech DAPS PCD jaCappella --methods Latent-Joint \
    --joint_views snac_32 dac_16 dac_44 --attacks snac_24 snac_44 encodec_48 dac_24 \
    --xfer_condition snac_24 --filecount 120 --out ../results_table2 --tag C1

# Table 3: DSP attacks
python benchmark.py --datasets AIR Freischuetz GuitarSet jaCappella LibriSpeech \
    --methods Latent-Cluster AudioSeal WavMark SilentCipher \
    --attacks gaussian amplitude lowpass resample --filecount 120 --out ../results_table3

# Figure 3 input: keep the audio
python benchmark.py --datasets Clotho LibriSpeech DAPS PCD jaCappella --attacks snac_24 --save_wavs --out ../results_audio
```

Optimization sets used in the paper:

| Paper | `--joint_views` |
| --- | --- |
| C1 | `snac_32 dac_16 dac_44` |
| C2 | `snac_32 encodec_24 encodec_32` |
| F1 | `snac_24 dac_24 encodec_24` |

### What one run does

For every dataset the script draws `--filecount` test files and `--calib_files` further clean files for calibration (disjoint when the dataset is large enough). Every file is cropped to its first `--seconds` seconds (default 5; `0` keeps full length). For each Latent-Mark method it estimates the null distribution of the projection on the calibration files, giving `mu`, `sigma`, `tau = mu + k sigma` and `alpha` per codec view (Eq. 6 and 8). Then for each test file and method it records:

- `clean_score` and `wm_score`: detector output on the clean and the watermarked clip. Detectability accuracy, TPR and FPR use `score > threshold`, where the threshold is 0 for Latent-Mark (normalized margin) and each baseline's published value.
- `attacked_score__<attack>`: detector output after the attack. Survivability is the fraction above the threshold.
- `clean_attacked_score__<attack>`: the same attack applied to the clean clip. Transferability is the fraction of files with a positive Delta-Score, `attacked_score - clean_attacked_score` (Section 3.3, Stage 4). With `--xfer_condition snac_24` it is computed only over files whose watermark survived SNAC 24 kHz, as in Section 5.2.

### Latent-Mark hyperparameters

Defaults follow Section 3. Every value is a flag.

| Flag | Default | Paper |
| --- | --- | --- |
| `--k` | 1.5 | calibration constant k in `tau = mu + k sigma` |
| `--gamma` | 1.5 | safety margin: the embedding target is `tau + gamma sigma` per view |
| `--steps`, `--lr` | 150, 0.005 | Adam steps on the waveform perturbation |
| `--beta`, `--sdr` | 2.5, 42 dB | budget `eps = clip(beta RMS(s) 10^(-SDR/20), eps_min, eps_max)` |
| `--eps_min`, `--eps_max` | 1e-4, 0.1 | |
| `--work_sr`, `--pad_multiple` | 44100, 4096 | working rate and padding for Latent-Joint (Stage 1) |
| `--calib_files`, `--calib_frames` | 42, 512 | clean files and sampled frames per file for the null distribution |
| `--target_mode` | `margin` | `tau` drops the safety margin; `gamma` uses an absolute target instead of a margin in null-std units |
| `--sigma_level` | `file` | std of the clip-mean projection over clean clips; `frame` uses individual frames |
| `--hinge` | `mean` | hinge on the clip-mean projection; `frame` applies it per frame |

For Latent-Joint each view's hinge is divided by its `alpha` (Eq. 9) and the detection score is the median of the per-view margins (Eq. 10).

### Output

```
<out>[_<tag>]/
├── config.json               every flag of the run
├── summary_all.csv           one row per dataset x method
└── <dataset>/
    ├── scores.csv            one row per file x method with every score above
    ├── summary.csv
    └── <method>/<stem>/      with --save_wavs: 1_original.wav, 2_watermarked.wav, 3_attacked.wav
                              (+ 3_attacked_<attack>.wav per attack, + analysis_plot.png with --plots)
```

## Tables

```bash
python summarize.py ../results_table1 ../results_table3            # rows = method, columns = dataset x metric
python summarize.py ../results_table2_C1 ../results_table2_C2 ../results_table2_F1 --by run --xfer
```

## Audio quality

See `../audio_quality_check/README.md`. The `--save_wavs` layout above is what `evaluate_quality.py --dir` scans.
