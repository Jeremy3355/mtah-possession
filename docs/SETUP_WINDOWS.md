# Running on Windows (WSL2) with an NVIDIA GPU

The pipeline is developed on macOS and Linux. On Windows, run it inside **WSL2 with Ubuntu**: the
bash scripts work unchanged, and CUDA in WSL2 uses the normal Windows NVIDIA driver.

## 1. One-time Windows setup (PowerShell as administrator)

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot if asked, open "Ubuntu", and create the Linux user.

- Install or update the **NVIDIA Windows driver** (a recent Game Ready or Studio driver for the RTX 50
  series).
- Do **not** install a Linux GPU driver inside WSL. `nvidia-smi` inside Ubuntu should already show
  the GPU.

## 2. Inside Ubuntu

```bash
sudo apt update && sudo apt install -y git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
(type -p gh >/dev/null) || sudo apt install -y gh
gh auth login
```

Keep everything under the Linux home (`~/MTAH2027`), not under `/mnt/c`. The Windows filesystem is
many times slower from WSL.

```bash
mkdir -p ~/MTAH2027 && cd ~/MTAH2027
gh repo clone Jeremy3355/mtah-possession
git clone https://github.com/metrica-sports/more-than-a-hack-2027
```

## 3. Competition data (not in any repository)

Copy the organisers' match bundle into `~/MTAH2027/more-than-a-hack-2027/data/` so that it holds
`tactical/` and, optionally, `broadcast/`. Copy it from the Windows side with:

```bash
cp -r /mnt/c/Users/<you>/<folder>/tactical ~/MTAH2027/more-than-a-hack-2027/data/
```

The data is not redistributable. Move it by USB or the local network, never through a cloud drive or
an online service.

## 4. Environment and reproduction of the submission

```bash
cd ~/MTAH2027/mtah-possession
uv python install 3.14.7
uv venv --python 3.14.7 .venv
uv pip install --python .venv/bin/python -r requirements.txt
bash scripts/download_dfl.sh                        # -> ~/MTAH2027/external_data/dfl, MD5-checked
.venv/bin/python scripts/calibrate_from_atd.py
git diff --stat calibration/                        # must be empty: same calibration as committed
.venv/bin/python scripts/make_submission.py --out submissions/v4.csv
md5sum submissions/v4.csv                           # must be 83fbddcecc98e7d10ed350e8f8c5683b
```

If the checksum differs, the pipeline is not reproducible across machines. That has to be fixed
before anything else.

## 5. GPU environment (phase 2, kept apart from the submission environment)

Deep-learning work uses its own environment so it can never change the submission's dependencies.

- **Python version:** 3.12, because PyTorch wheels may lag Python 3.14.
- **PyTorch build:** one built for CUDA 12.8 or newer. The RTX 50 series (Blackwell) needs it.

```bash
uv venv --python 3.12 .venv-gpu
uv pip install --python .venv-gpu/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv-gpu/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
