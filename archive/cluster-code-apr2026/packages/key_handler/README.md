# key_handler

Centralized API key management. Put your keys in one place; scripts and Claude access them without hardcoding secrets.

## Setup

1. Copy the template to your actual key file:

   ```bash
   cp packages/key_handler/key_handler/key_handler__template.py \
      packages/key_handler/key_handler/key_handler.py
   ```

2. Open `key_handler.py` and fill in your keys:

   ```python
   class KeyHandler:
       openai_key: str = "sk-..."
       anthropic_key: str = "sk-ant-..."
       hf_key: str = "hf_..."
       # ...
   ```

3. `key_handler.py` is listed in `.gitignore` — it will never be committed.

4. Call `KeyHandler.set_env_key()` at the top of any script that needs API access:

   ```python
   from key_handler import KeyHandler
   KeyHandler.set_env_key()  # injects keys into os.environ

   import openai  # now picks up OPENAI_API_KEY automatically
   ```

## Keys managed

| Attribute        | Environment variable(s)          |
|-----------------|----------------------------------|
| `openai_key`     | `OPENAI_API_KEY`                 |
| `anthropic_key`  | `ANTHROPIC_API_KEY`              |
| `hf_key`         | `HF_TOKEN`, `HF_API_KEY`         |
| `together_key`   | `TOGETHER_API_KEY`               |
| `openrouter_key` | `OPENROUTER_API_KEY`             |
| `runpod_key`     | `RUNPOD_API_KEY`                 |
| `wandb_key`      | `WANDB_API_KEY`                  |

Keys that still contain the placeholder value (`your-...`) or are empty are silently skipped.

## Install

```bash
pip install -e packages/key_handler/
```
