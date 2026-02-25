# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This is a **Medical AI Agent ART Reinforcement Learning Training System** — a Python ML training pipeline using the ART (Automatic Reward Training) framework by OpenPipe. It trains a medical Q&A agent via reinforcement learning for the Kaggle Cure-Bench competition.

**Key files:**
- `med_art_rl.py` — Main training script (contains Colab-specific syntax; see caveats below)
- `inference.py` — Inference testing module (valid standard Python, imports from `med_art_rl`)
- `config.yaml.template` → copy to `config.yaml` and fill in API keys before running
- `requirements.txt` — Python dependencies (note: `sqlite3` entry is a built-in, must be skipped during pip install)

### Important Caveats

1. **Colab-specific syntax in `med_art_rl.py`**: Lines 81–87 use Jupyter `!` magic commands (`!uv pip install`, `!pip uninstall`), and lines 109–111 import `google.colab` unconditionally. This makes `med_art_rl.py` unparseable as standard Python. The file was originally a Colab notebook.

2. **`requirements.txt` contains `sqlite3`**: This is a Python built-in module and will cause `pip install -r requirements.txt` to fail. Install dependencies with:
   ```bash
   pip install $(grep -v '^#' requirements.txt | grep -v '^$' | grep -v 'sqlite3' | tr '\n' ' ')
   ```

3. **GPU required for training/inference**: The project uses vLLM, PyTorch, and triton which all require NVIDIA GPU with CUDA. Without GPU, only non-ML components (config, database, data models) can be tested.

4. **API keys required**: Full execution requires `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and `hf_token` configured in `config.yaml`. Without these, the main training script will raise `ValueError` at startup.

### Lint / Test / Build / Run

- **Lint**: `flake8 inference.py --max-line-length=120` (only `inference.py` is valid standard Python; `med_art_rl.py` has Colab syntax errors)
- **Syntax check**: `python3 -c "import ast; ast.parse(open('inference.py').read())"`
- **Format check**: `black --check inference.py`
- **Run training** (requires GPU + API keys + Colab env): `python med_art_rl.py`
- **Run inference** (requires GPU + API keys + trained model): `python inference.py`
- **Config setup**: `cp config.yaml.template config.yaml` then edit with your API keys
