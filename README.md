# Latent-Mark: An Audio Watermark Robust to Neural Codec Compression

**Accepted to Interspeech 2026!**

[![arXiv](https://img.shields.io/badge/arXiv-2603.05310-b31b1b.svg)](https://arxiv.org/abs/2603.05310)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Yen-Shan Chen\*, Shih-Yu Lai\*, Ying-Jung Tsou, Yi-Cheng Lin, Bing-Yu Chen, Yun-Nung Chen, Hung-yi Lee, Shang-Tse Chen
(\* equal contribution)

Latent-Mark is a zero-bit audio watermark designed to survive neural codec compression. Instead of adding an imperceptible waveform pattern, it optimizes the audio so that its encoded latent representation shifts along a secret axis in the codec's latent space. Cross-Codec Optimization jointly optimizes the same perturbation across several surrogate codecs so the mark transfers to unseen codecs.

## Repository layout

```
setup_env.sh                       creates the conda environment
watermark_research/
  src/latentmark.py                the method: codec views, secret axes, calibration, embed / detect, baselines, attacks
  src/benchmark.py                 detectability, survivability and transferability experiments
  src/summarize.py                 paper-style tables from result folders
  legacy/                          original per-experiment scripts
  README.md                        full usage
audio_quality_check/               SNR, LSD, PESQ, STOI, UTMOS, SI-SNR and the Figure 3 plots
raw_bench/                         git submodule: RAW-Bench datasets and baseline checkpoints
```

## Installation

```bash
git clone --recurse-submodules git@github.com:yenshan0530/Latent-Mark.git
cd Latent-Mark
bash setup_env.sh          # conda env "latentmark": python 3.10, torch 2.1 (CUDA 12.1), codecs, baselines
conda activate latentmark
```

`CUDA=cpu bash setup_env.sh` installs CPU-only torch. Codec weights and baseline checkpoints download automatically on first use. Datasets: follow the `raw_bench` README, then place them under `dataset/` with one subfolder per dataset.

## Quick start

```python
import latentmark as lm                       # from watermark_research/src
wm = lm.LatentMark(["snac_24"], "cuda")       # Latent-Cluster on SNAC 24 kHz
wm.calibrate(clean_files)                     # null distribution on ~40 clean clips
wav, sr = lm.load_audio("speech.wav")
marked, _ = wm.embed(wav, sr)                 # (1, T) at 24 kHz
print(wm.detect(marked, 24000) > 0)           # True: normalized margin above the calibrated threshold
attacked = lm.make_attack("encodec_48", "cuda")(marked, 24000)
print(wm.detect(attacked, 24000) > 0)
```

Benchmark on one dataset:

```bash
cd watermark_research/src
python benchmark.py --datasets LibriSpeech --methods Latent-Cluster Latent-Joint AudioSeal \
    --attacks snac_24 snac_44 encodec_48 dac_24 --filecount 120
```

See `watermark_research/README.md` for every flag and the exact commands behind each table.

## Method names

| Paper | Code |
| --- | --- |
| Latent-Cluster | `Latent-Cluster` (`--single_view snac_24`) |
| Latent-PCA | `Latent-PCA` |
| Latent-Random | `Latent-Random` |
| Latent-Joint, set C1 / C2 / F1 | `Latent-Joint` with `--joint_views snac_32 dac_16 dac_44` / `snac_32 encodec_24 encodec_32` / `snac_24 dac_24 encodec_24` |

## Reproducing the paper's experiments

| Experiment | Command |
| --- | --- |
| Table 1: detectability and survivability under SNAC | `benchmark.py --attacks snac_24` with all seven methods |
| Table 2: transfer to unseen codecs per optimization set | `benchmark.py --methods Latent-Joint --joint_views ... --attacks snac_24 snac_44 encodec_48 dac_24 --xfer_condition snac_24 --tag C1` |
| Table 3: DSP attacks | `benchmark.py --attacks gaussian amplitude lowpass resample` |
| Figure 3: audio quality | `benchmark.py --save_wavs`, then `audio_quality_check/evaluate_quality.py` and `plot_for_paper.py` |

## Citation

```bibtex
@inproceedings{chen2026latentmark,
  title     = {Latent-Mark: An Audio Watermark Robust to Neural Codec Compression},
  author    = {Chen, Yen-Shan and Lai, Shih-Yu and Tsou, Ying-Jung and Lin, Yi-Cheng and Chen, Bing-Yu and Chen, Yun-Nung and Lee, Hung-yi and Chen, Shang-Tse},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026}
}
```

## Acknowledgment

This work was supported in part by the National Science and Technology Council under Grants NSTC 114-2634-F-002-004 and NSTC 114-2634-F-002-003-MBK.

## License

MIT. See [LICENSE](LICENSE).
