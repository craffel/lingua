# Copyright (c) Meta Platforms, Inc. and affiliates.
# Save an initial (randomly-initialized) checkpoint from a training config
# and optionally push it to the HuggingFace Hub.
#
# Usage:
#   python -m apps.main.save_init_ckpt config=<path/to/train_config.yaml> \
#       [dump_dir=<out_dir>] [hf_repo=<owner/repo>]

import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from apps.main.train import TrainArgs
from apps.main.transformer import LMTransformer
from lingua.args import dump_config
from lingua.checkpoint import CONFIG_NAME, CONSOLIDATE_FOLDER, CONSOLIDATE_NAME
from lingua.tokenizer import build_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def save_init_ckpt(args: TrainArgs, hf_repo: str | None = None):
    tokenizer = build_tokenizer(
        args.data.tokenizer.name,
        args.data.tokenizer.path,
        args.data.tokenizer.tokenizers,
        args.data.tokenizer.dropout,
        superset_code_name=args.data.tokenizer.superset_code_name,
        n_words=args.data.tokenizer.n_words,
    )
    n_words: int = getattr(tokenizer, "n_words")

    # validate_train_args checks distributed world size; handle only what we need.
    if args.model.vocab_size < 0:
        logger.info(f"Setting model vocab size to {n_words}")
        args.model.vocab_size = n_words
    assert args.model.vocab_size == n_words, (
        f"Config vocab_size {args.model.vocab_size} != tokenizer n_words {n_words}"
    )
    assert args.dump_dir, "dump_dir must be set"

    out_dir = Path(args.dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.model.max_seqlen = args.data.seq_len

    logger.info("Building model …")
    model = LMTransformer(args.model)

    logger.info(f"Initializing weights with seed {args.model.seed}")
    with torch.random.fork_rng():
        torch.manual_seed(args.model.seed)
        model.init_weights()

    # Save a consolidated (single-file) checkpoint
    consolidated_dir = out_dir / CONSOLIDATE_FOLDER
    consolidated_dir.mkdir(exist_ok=True)
    ckpt_path = consolidated_dir / CONSOLIDATE_NAME
    config_path = consolidated_dir / CONFIG_NAME

    logger.info(f"Saving checkpoint to {ckpt_path}")
    torch.save({"model": model.state_dict()}, ckpt_path)
    config_path.write_text(
        json.dumps(
            OmegaConf.to_container(OmegaConf.structured(args), resolve=True), indent=2
        )
    )
    dump_config(args, out_dir / "config.yaml")
    logger.info("Checkpoint saved.")

    if hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        logger.info(f"Uploading checkpoint to HuggingFace repo: {hf_repo}")
        api.upload_folder(
            repo_id=hf_repo,
            folder_path=str(consolidated_dir),
            repo_type="model",
        )
        logger.info("Upload complete.")

    return ckpt_path


def main():
    cli_args = OmegaConf.from_cli()

    # Pull out non-TrainArgs keys before merging
    hf_repo: str | None = cli_args.pop("hf_repo", None)

    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    args: TrainArgs = OmegaConf.to_object(cfg)  # type: ignore[assignment]

    save_init_ckpt(args, hf_repo=hf_repo)


if __name__ == "__main__":
    main()
