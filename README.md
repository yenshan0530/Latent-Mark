# Latent-Mark: An Audio Watermark Robust to Neural Codec Compression

**Accepted to Interspeech 2026!**

[![arXiv](https://img.shields.io/badge/arXiv-2603.05310-b31b1b.svg)](https://arxiv.org/abs/2603.05310)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Yen-Shan Chen\*, Shih-Yu Lai\*, Ying-Jung Tsou, Yi-Cheng Lin, Bing-Yu Chen, Yun-Nung Chen, Hung-yi Lee, Shang-Tse Chen
(\* equal contribution)

Latent-Mark is a zero-bit audio watermark designed to survive neural codec compression. Instead of adding an imperceptible waveform pattern, it optimizes the audio so that its encoded latent representation shifts along a secret axis in the codec's latent space. Cross-Codec Optimization jointly optimizes the same perturbation across several surrogate codecs so the mark transfers to unseen codecs.

## Repository layout

```
watermark_research/      Watermark embedding, detection, and codec-attack benchmarks
  src/
    watermark_testing.py            Single-codec benchmark (SNAC 24 kHz attack)
    transferbility_testing.py       Cross-codec optimization with a selectable attack codec
    transferbility_testing_all.py   Sweep over optimization sets x attack codecs in one run
    watermark_against_attacks.py    Robustness to DSP attacks (noise, gain, low-pass, resampling)
    summarize_results.py            Aggregate detectability / survivability into one table
    summarize_sur_det.py            Same, alternate layout
    generate_summary.py             Pass-rate table by optimization set x attack codec
  README.md                         Full usage for every script
audio_quality_check/     SNR, LSD, PESQ, STOI, UTMOS, SI-SNR evaluation and plots
raw_bench/               Git submodule: RAW-Bench datasets, baseline checkpoints, attack backends
```

## Installation

```bash
git clone --recurse-submodules git@github.com:yenshan0530/Latent-Mark.git
cd Latent-Mark
conda env create -f watermark_research/environment.yml
conda activate aw
```

If you cloned without `--recurse-submodules`, run `git submodule update --init --recursive`. The `raw_bench` submodule provides the evaluation datasets and the AudioSeal, WavMark, and SilentCipher checkpoints. Follow its README to download the datasets, then place them under `dataset/` at the repository root with one subfolder per dataset (`dataset/LibriSpeech`, `dataset/DAPS`, ...).

Codec weights (SNAC, EnCodec, DAC) are downloaded automatically from Hugging Face on first use.

## Quick start

Embed, attack with SNAC 24 kHz, and detect on one dataset:

```bash
cd watermark_research/src
python watermark_testing.py --mode both --datasets LibriSpeech \
  --watermarks SemanticCluster SemanticPCA SemanticRandom --filecount 120
```

Cross-codec optimization across SNAC, DAC 16 kHz, and DAC 44 kHz, attacked with every unseen codec:

```bash
python transferbility_testing_all.py --datasets LibriSpeech --opt_set Opt_B1 --attack all --filecount 120
```

See `watermark_research/README.md` for every flag, output layout, and the DSP and quality pipelines.

## Method names

| Paper | Code |
| --- | --- |
| Latent-Cluster | `SemanticCluster` |
| Latent-PCA | `SemanticPCA` |
| Latent-Random | `SemanticRandom` |
| Latent-Joint | `JointManifold` |

## Reproducing the paper's experiments

| Experiment | Script | Key flags |
| --- | --- | --- |
| Table 1: detectability and survivability under SNAC | `watermark_testing.py` | `--mode both` |
| Table 2, sets C1 / C2 / F1 | `transferbility_testing_all.py` | `--opt_set Opt_B1` / `Opt_B2` / `Opt_Mix` |
| Table 2, sets D1 / D2 | `transferbility_testing.py` | `--joint_codecs snac dac44 funcodec` / `snac funcodec apcodec` |
| Table 2, single-codec rows | `transferbility_testing_all.py` | `--watermarks SemanticCluster SemanticPCA SemanticRandom` |
| Table 3: DSP attacks | `watermark_against_attacks.py` | `--mode benchmark` |
| Figure 3: audio quality | `audio_quality_check/` | `evaluate_quality.py`, then `plot_for_paper.py` |

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
