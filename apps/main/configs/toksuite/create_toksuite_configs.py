from pathlib import Path

import yaml

# python apps/main/configs/toksuite/create_toksuite_configs.py
models=("google-gemma-2-2b" "common-pile-comma-v0.1" "meta-llama-Llama-3.2-1B" "microsoft-Phi-3-mini-4k-instruct" "gpt2" "bigscience-bloom" "facebook-xglm-564M" "mistralai-tekken" "google-byt5-small" "google-bert-bert-base-multilingual-cased" "Qwen-Qwen3-8B" "tokenmonster-englishcode-32000-consistent-v1" "tiktoken-gpt-4o" "CohereLabs-aya-expanse-8b")
tokenizers=("google/gemma-2-2b" "common-pile/comma-v0.1-1t" "meta-llama/Llama-3.2-1B" "microsoft/Phi-3-mini-4k-instruct" "gpt2" "bigscience/bloom" "facebook/xglm-564M" "mistralai/tekken" "google/byt5-small" "google-bert/bert-base-multilingual-cased" "Qwen/Qwen3-8B" "tokenmonster/englishcode-32000-consistent-v1" "tiktoken/gpt-4o" "CohereLabs/aya-expanse-8b")
TOKENIZER_CONFIGS = [
    {
        "name": "huggingface",
        "path": "google/gemma-2-2b"
    },
    {
        "name": "huggingface",
        "path": "common-pile/comma-v0.1-1t"
    },
    {
        "name": "huggingface",
        "path": "meta-llama/Llama-3.2-1B"
    },
    {
        "name": "huggingface",
        "path": "microsoft/Phi-3-mini-4k-instruct"
    },
    {
        "name": "huggingface",
        "path": "gpt2"
    },
    {
        "name": "huggingface",
        "path": "bigscience/bloom"
    },
    {
        "name": "huggingface",
        "path": "facebook/xglm-564M"
    },
    {
        "name": "tekken",
        "path": "mistralai/tekken"
    },
    {
        "name": "huggingface",
        "path": "google/byt5-small"
    },
    {
        "name": "huggingface",
        "path": "google-bert/bert-base-multilingual-cased"
    },
    {
        "name": "huggingface",
        "path": "Qwen/Qwen3-8B"
    },
    {
        "name": "tokenmonster",
        "path": "tokenmonster/englishcode-32000-consistent-v1"
    },
    {
        "name": "tiktoken",
        "path": "tiktoken/gpt-4o"
    },
    {
        "name": "huggingface",
        "path": "CohereLabs/aya-expanse-8b"
    }
]
if __name__ == "__main__":
    config_dir = Path(__file__).parent
    for tok_config in TOKENIZER_CONFIGS:
        model = tok_config["path"].replace("/", "-")
        name = tok_config["name"]
        path = tok_config["path"]
        config = {
            "name": f"toksuite_{model}",
            "dump_dir": f"/fsx/craffel/toksuite/lingua_logs/{model}/",
            "data": {
                "tokenizer": {
                    "name": name,
                    "path": path
                    }
            },
            "checkpoint": {
               "path": f"/fsx/craffel/toksuite/lingua_logs/{model}/checkpoints"
            }
        }
        with open(config_dir / f"{model}.yaml", "w") as f:
            yaml.dump(config, f)