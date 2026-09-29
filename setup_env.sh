#!/usr/bin/env bash
# Creates the `latentmark` conda environment used by every script in this repository.
#
#   bash setup_env.sh            # CUDA 12.1 wheels (default)
#   CUDA=cpu bash setup_env.sh   # CPU-only torch
#
# Notes
#   * audiocraft pins av==11.0.0, which has no binary wheel on PyPI, so av comes from conda-forge.
#   * silentcipher declares torch<=2.0 but runs on torch 2.1; it is installed without dependencies.
set -euo pipefail

ENV_NAME="${ENV_NAME:-latentmark}"
CUDA="${CUDA:-cu121}"
PY="$(conda info --base)/envs/${ENV_NAME}/bin/python"

conda create -y -n "${ENV_NAME}" python=3.10
conda install -y -n "${ENV_NAME}" -c conda-forge "av=11.0.0" "numpy=1.26.4"

"${PY}" -m pip install --upgrade pip setuptools wheel
if [ "${CUDA}" = "cpu" ]; then
  "${PY}" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0
else
  "${PY}" -m pip install --extra-index-url "https://download.pytorch.org/whl/${CUDA}" \
      torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 xformers==0.0.22.post7
fi
"${PY}" -m pip install -r watermark_research/requirements.txt
"${PY}" -m pip install --no-deps silentcipher==1.0.5

"${PY}" - <<'EOF'
import torch, torchaudio, audiocraft, snac, dac, audioseal, wavmark, silentcipher, av
print(f"OK  torch {torch.__version__}  torchaudio {torchaudio.__version__}  cuda={torch.cuda.is_available()}")
EOF
echo "Activate with:  conda activate ${ENV_NAME}"
