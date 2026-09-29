# Legacy experiment scripts

These are the original scripts used while developing Latent-Mark, kept for reference. Each one carries its own
copy of the watermarkers and attacks and its own command-line interface. The maintained pipeline is
`../src/benchmark.py` on top of `../src/latentmark.py`, which covers everything here with one interface.

| Script | What it did |
| --- | --- |
| `watermark_testing.py` | Single-codec benchmark: embed, SNAC 24 kHz round trip, detect |
| `transferbility_testing.py` | Cross-codec optimization with a selectable attack codec |
| `transferbility_testing_all.py` | Sweep of optimization sets x attack codecs |
| `watermark_against_attacks.py` | DSP attacks: Gaussian noise, gain, low-pass, resampling |
| `summarize_results.py`, `summarize_sur_det.py`, `generate_summary.py` | Table builders for the outputs above |

Run them from this folder. They expect datasets under `../../dataset` unless told otherwise with `--base_dir`.
