# Copyright (c) Meta Platforms, Inc. and affiliates.

import gc
import json
import logging
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.distributed._tensor import DTensor

from apps.main.train import TrainState, every_n_steps, set_preemption_flag
from apps.main.transformer import (
    LMTransformer,
    LMTransformerArgs,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token,
    tp_parallelize,
)
from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import (
    CheckpointArgs,
    CheckpointManager,
    consolidate_checkpoints,
    load_from_checkpoint,
)
from lingua.data import TRAIN_DATA_FILE_PATTERN, distribute_data_to_rank, loop_on_jsonl
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    check_model_value_range,
    dist_mean_dict,
    get_device_mesh,
    get_global_rank,
    get_is_master,
    get_world_size,
    init_signal_handler,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
)
from lingua.logger import init_logger
from lingua.metrics import LoggingArgs, MetricLogger, get_num_params
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs
from lingua.tokenizer import TokenizerArgs, build_token_bytes, build_tokenizer

logger = logging.getLogger()

DUMP_DOCS = os.environ.get("DUMP_DOCS", "False") == "True"
DUMP_DOCS_MAX_SAMPLES = int(os.environ.get("DUMP_DOCS_MAX_SAMPLES", "2"))
# DUMP_DIR = os.environ.get("DUMP_DIR", "/scratch/gsa/train/dump-mod")

@dataclass
class QADataArgs:
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 64
    seq_len: int = 64
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    file_pattern: str = TRAIN_DATA_FILE_PATTERN
    text_key: str = "text"
    question_key: str = "question"
    answer_key: str = "answer"
    n_views: int = 2
    prefetch_size: int = 64
    load_async: bool = True
    suitable_tokenizer_key: str = "suitable_tokenizer"
    suitable_tokenizer_probability: float = 0.0
    suitable_tokenizer_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainAnswerOnlyArgs:
    name: str = "lingua-answer-only"
    dump_dir: str = ""
    seed: int = 42
    grad_acc_steps: int = 1
    steps: int = 1000

    data: QADataArgs = field(default_factory=QADataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)
    track_source_metrics: bool = False
    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None

def _source_iterators(args: TrainAnswerOnlyArgs):
    rank = get_global_rank()
    world_size = get_world_size()
    per_source = {}
    for source in args.data.sources:
        source_path = os.path.join(args.data.root_dir, source)
        state = distribute_data_to_rank(
            source_path,
            rank=rank,
            world_size=world_size,
            file_pattern=args.data.file_pattern,
        )
        per_source[source] = loop_on_jsonl(
            state["file_path"],
            state["position"],
            state["block_size"],
            state["offset"],
            state["current_iter"],
        )
    return per_source


def _normalize_weights(sources: Dict[str, float]) -> np.ndarray:
    weights = np.array([float(v) for v in sources.values()], dtype=np.float64)
    weights = weights / weights.sum()
    return weights


def _to_list(token_ids):
    if isinstance(token_ids, np.ndarray):
        return token_ids.tolist()
    return list(token_ids)


def _build_example(
    row: Dict[str, Any],
    tokenizer,
    seq_len: int,
    add_bos: bool,
    add_eos: bool,
    text_key: str,
    question_key: str,
    answer_key: str,
    tokenizer_choice: Optional[int] = None,
):
    full_text = row.get(text_key)
    if full_text is None:
        full_text = row.get("content")
    question = row.get(question_key)
    answer = row.get(answer_key)

    if full_text is None:
        return None

    if question is not None and answer is not None:
        question_text = str(question)
        answer_text = str(answer)
        full_text = f"{question_text}{answer_text}"
        prompt_text = question_text
    else:
        if question is None:
            parts = str(full_text).rsplit(" ", 1)
            question = (parts[0] + " ") if len(parts) > 1 else str(full_text)
        prompt_text = str(question)
    encode_kwargs: Dict[str, Any] = {"add_bos": add_bos, "add_eos": add_eos}
    prompt_encode_kwargs: Dict[str, Any] = {"add_bos": add_bos, "add_eos": False}
    if tokenizer_choice is not None:
        encode_kwargs["tokenizer_choice"] = tokenizer_choice
        prompt_encode_kwargs["tokenizer_choice"] = tokenizer_choice

    full_ids = _to_list(tokenizer.encode(str(full_text), **encode_kwargs))
    prompt_ids = _to_list(tokenizer.encode(prompt_text, **prompt_encode_kwargs))

    full_ids = full_ids[: seq_len + 1]
    if len(full_ids) < 2:
        return None

    input_ids = full_ids[:-1]
    labels = full_ids[1:]

    prompt_target_count = max(0, min(len(labels), len(prompt_ids) - 1))
    for i in range(prompt_target_count):
        labels[i] = -100

    # import code; code.interact(local=locals()|globals())

    pad_id = getattr(tokenizer, "pad_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_id", 0)
    pad_id = int(pad_id)

    if len(input_ids) < seq_len:
        pad_n = seq_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_n
        labels = labels + [-100] * pad_n
    else:
        input_ids = input_ids[:seq_len]
        labels = labels[:seq_len]

    return input_ids, labels


def _sample_row_tokenizer_choice(
    row: Dict[str, Any],
    tokenizer,
    suitable_tokenizer_key: str,
    suitable_tokenizer_probability: float,
    suitable_tokenizer_map: Dict[str, str],
) -> Optional[int]:
    if not hasattr(tokenizer, "sample_tokenizer"):
        return None

    preferred_tokenizer = _map_dataset_tokenizer_to_superset_key(
        dataset_tokenizer_name=row.get(suitable_tokenizer_key),
        tokenizer=tokenizer,
        suitable_tokenizer_map=suitable_tokenizer_map,
    )

    try:
        if preferred_tokenizer == "random":
            sampled_choice, _ = tokenizer.sample_tokenizer()
        else:
            sampled_choice, _ = tokenizer.sample_tokenizer(
            preferred_tokenizer=preferred_tokenizer,
            preferred_probability=suitable_tokenizer_probability,
        )
        return int(sampled_choice)
    except TypeError:
        sampled_choice, _ = tokenizer.sample_tokenizer()
        return int(sampled_choice)
    except Exception:
        return None


def _map_dataset_tokenizer_to_superset_key(
    dataset_tokenizer_name: Any,
    tokenizer,
    suitable_tokenizer_map: Dict[str, str],
) -> Optional[str]:
    if dataset_tokenizer_name is None:
        return None

    raw_name = str(dataset_tokenizer_name).strip()
    if raw_name == "":
        return None

    lowered_name = raw_name.lower()

    mapped = suitable_tokenizer_map.get(raw_name)
    if mapped is None:
        mapped = suitable_tokenizer_map.get(lowered_name)

    candidate_name = mapped if mapped is not None else raw_name

    if not hasattr(tokenizer, "tokenizers") or not isinstance(tokenizer.tokenizers, dict):
        return candidate_name

    tokenizer_keys = list(tokenizer.tokenizers.keys())
    if candidate_name in tokenizer.tokenizers:
        return candidate_name

    lowered_candidate = candidate_name.lower()
    for key in tokenizer_keys:
        lowered_key = key.lower()
        if lowered_key == lowered_candidate:
            return key

    for key in tokenizer_keys:
        lowered_key = key.lower()
        if lowered_key.endswith(f"/{lowered_candidate}"):
            return key
        if lowered_candidate.endswith(f"/{lowered_key}"):
            return key

    return candidate_name


def _batch_iterator(args: TrainAnswerOnlyArgs, tokenizer):
    source_names = list(args.data.sources.keys())
    source_weights = _normalize_weights(args.data.sources)
    source_to_id = {name: idx for idx, name in enumerate(source_names)}
    source_iters = _source_iterators(args)
    rng = np.random.default_rng((args.data.seed, get_global_rank(), get_world_size()))

    try:
        while True:
            batch_inputs = []
            batch_labels = []
            batch_source_ids = []
            while len(batch_inputs) < args.data.batch_size:
                source = source_names[rng.choice(len(source_names), p=source_weights)]
                row, _ = next(source_iters[source])
                tokenizer_choice = _sample_row_tokenizer_choice(
                    row=row,
                    tokenizer=tokenizer,
                    suitable_tokenizer_key=args.data.suitable_tokenizer_key,
                    suitable_tokenizer_probability=args.data.suitable_tokenizer_probability,
                    suitable_tokenizer_map=args.data.suitable_tokenizer_map,
                )
                example = _build_example(
                    row=row,
                    tokenizer=tokenizer,
                    seq_len=args.data.seq_len,
                    add_bos=args.data.add_bos,
                    add_eos=args.data.add_eos,
                    text_key=args.data.text_key,
                    question_key=args.data.question_key,
                    answer_key=args.data.answer_key,
                    tokenizer_choice=tokenizer_choice,
                )
                if example is None:
                    continue
                x, y = example
                batch_inputs.append(x)
                batch_labels.append(y)
                batch_source_ids.append(source_to_id[source])

            yield (
                torch.tensor(batch_inputs, dtype=torch.long),
                torch.tensor(batch_labels, dtype=torch.long),
                torch.tensor(batch_source_ids, dtype=torch.long),
            )
    finally:
        for it in source_iters.values():
            it.close()


def maybe_dump_training_batch(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    source_ids: torch.Tensor,
    source_names: list[str],
    step: int,
    acc_step: int,
    dump_dir: str,
) -> None:
    if not DUMP_DOCS:
        return

    effective_dump_dir = dump_dir 
    if not effective_dump_dir:
        return

    rank = get_global_rank()
    dump_path = Path(effective_dump_dir) / "training_docs" / f"rank_{rank}.jsonl"
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    cpu_input_ids = input_ids.detach().cpu()
    cpu_labels = labels.detach().cpu()
    cpu_source_ids = source_ids.detach().cpu()
    n_samples = min(DUMP_DOCS_MAX_SAMPLES, cpu_input_ids.shape[0])

    with open(dump_path, "a") as f_dump:
        for sample_idx in range(n_samples):
            input_id_list = [x for x in cpu_input_ids[sample_idx].tolist() if x != 2]
            label_id_list = [x for x in cpu_labels[sample_idx].tolist() if x != 2 and x != -100]
            label_id_valid = [token_id for token_id in label_id_list if token_id != -100]
            source_id = int(cpu_source_ids[sample_idx].item())

            row = {
                "step": int(step),
                "acc_step": int(acc_step),
                "sample_idx": int(sample_idx),
                "source_id": source_id,
                "source_name": source_names[source_id] if 0 <= source_id < len(source_names) else None,
                "input_ids": input_id_list,
                "label_ids": label_id_list,
            }

            try:
                row["input_text"] = tokenizer.decode(input_id_list, skip_special_tokens=False)
            except TypeError:
                try:
                    row["input_text"] = tokenizer.decode(input_id_list)
                except Exception:
                    row["input_text"] = None
            except Exception:
                row["input_text"] = None

            try:
                row["label_text"] = tokenizer.decode(label_id_valid, skip_special_tokens=False)
            except TypeError:
                try:
                    row["label_text"] = tokenizer.decode(label_id_valid)
                except Exception:
                    row["label_text"] = None
            except Exception:
                row["label_text"] = None

            json.dump(row, f_dump)
            f_dump.write("\n")



def validate_train_args(args: TrainAnswerOnlyArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    assert (
        args.model.vocab_size == output_size
    ), "Vocab size should be the same as output size"

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


preemption_flag = dict(flag=False)

def train(args: TrainAnswerOnlyArgs):
    with ExitStack() as context_stack:
        assert args.dump_dir, "dump_dir is required"
        assert args.data.root_dir is not None, "data.root_dir is required"
        assert len(args.data.sources) > 0, "data.sources must be non-empty"

        tokenizer = build_tokenizer(
            args.data.tokenizer.name,
            args.data.tokenizer.path,
            args.data.tokenizer.tokenizers,
            args.data.tokenizer.dropout,
            superset_code_name=args.data.tokenizer.superset_code_name,
            n_words=args.data.tokenizer.n_words,
        )
        validate_train_args(
            args,
            tokenizer.n_words,
        )
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(str(Path(args.dump_dir) / "train.log"))
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                "No CUDA GPUs are visible before distributed init. "
                f"cuda_available={torch.cuda.is_available()} "
                f"device_count={torch.cuda.device_count()} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')} "
                f"SLURM_NODELIST={os.environ.get('SLURM_JOB_NODELIST')} "
                f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}. "
                "Ensure the job is launched on a GPU allocation (e.g. launcher=sbatch with ngpu>=1), "
                "and verify inside the job with `nvidia-smi`."
            )

        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        source_names = list(args.data.sources.keys())
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")
        


        torch.manual_seed(args.seed)
        logger.info("Building model")
        if args.model.vocab_size < 0:
            args.model.vocab_size = tokenizer.n_words
        assert args.model.vocab_size == tokenizer.n_words, "model.vocab_size must match tokenizer.n_words"
        args.model.max_seqlen = args.data.seq_len
        
        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        use_bf16_autocast = str(args.distributed.model_dtype).lower() in {"bf16", "bfloat16"}

        if args.checkpoint.init_ckpt_path:
            # todo: maybe auto load the largest ckpt
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            if args.checkpoint.load_init_optimizer_state:
                load_from_checkpoint(args.checkpoint.init_ckpt_path, model, optimizer, model_key="model") # Put model_key="" if its directly the model checkpoint
            else:
                load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
            model.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)
        
        data_iter = _batch_iterator(args, tokenizer)

        token_bytes_dict = build_token_bytes(tokenizer, tokenizer.n_words)
        token_bytes_tensor = torch.zeros(tokenizer.n_words, dtype=torch.int64, device="cuda")
        for tid, nb in token_bytes_dict.items():
            token_bytes_tensor[tid] = nb

        data_loader_state = {
            "start_token": 0,
            "it_state": {},
            "output_seq_len": args.data.seq_len,
            "n_views": args.data.n_views,
            "seq_len": 0,
        }

        log_freq = 10
        if args.logging is not None and getattr(args.logging, "freq", None) is not None:
            log_freq = int(args.logging.freq)

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        if args.checkpoint.save_init_ckpt:
            if checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                ):
                _ = consolidate_checkpoints(str(checkpoint.existing_saves[-1]))

        n_tokens = 0
        t_last = timer()

        gc.disable()
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        saved = False
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            input_ids, labels, source_ids, token_bytes = next(data_iter)
            maybe_dump_training_batch(
                tokenizer,
                input_ids,
                labels,
                source_ids,
                source_names,
                train_state.step,
                train_state.acc_step,
                args.dump_dir,
            )
            data_load_time = round(timer() - data_load_start, 4)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            input_ids = input_ids.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            source_ids = source_ids.cuda(non_blocking=True)

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
                logits = model(input_ids)
                token_losses = F.cross_entropy(
                    logits.flatten(end_dim=-2).float(),
                    labels.flatten(end_dim=-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(labels)
            valid_token_mask = labels != -100
            loss = token_losses[valid_token_mask].mean() if valid_token_mask.any() else token_losses.new_zeros(())
            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            ## Accuracy calculation (for logging only, not used for training)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid = valid_token_mask
                denom = valid.sum().clamp_min(1)
                corrects = ((preds == labels) & valid).float().sum()
                token_count = denom.float()

                # BPB: sum nats and bytes only over valid (non-ignored) tokens
                y1d = labels.reshape(-1)
                valid1d = valid.reshape(-1)
                ysafe = torch.where(valid1d, y1d, torch.zeros_like(y1d))
                nb = torch.where(valid1d, token_bytes_tensor[ysafe], torch.zeros_like(y1d))
                counted = nb > 0
                bpb_nats_sum = token_losses.reshape(-1)[counted].sum()
                bpb_bytes_sum = nb[counted].sum()

                source_stats = None
                if args.track_source_metrics:
                    source_stats = {}
                    for source_id, source_name in enumerate(source_names):
                        sample_mask = source_ids == source_id
                        source_token_mask = valid & sample_mask.unsqueeze(1)
                        source_token_count = source_token_mask.sum()
                        if source_token_count.item() > 0:
                            source_loss_sum = token_losses[source_token_mask].sum()
                            source_corrects = ((preds == labels) & source_token_mask).float().sum()
                        else:
                            source_loss_sum = token_losses.new_zeros(())
                            source_corrects = token_losses.new_zeros(())
                        source_stats[source_name] = {
                            "loss_sum": source_loss_sum,
                            "token_count": source_token_count.float(),
                            "corrects": source_corrects,
                        }

            # optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # n_tokens += int((labels != -100).sum().item())

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                to_sync["corrects/out"] = corrects.item()
                to_sync["token_count/out"] = token_count.item()

                if args.track_source_metrics and source_stats is not None:
                    for source_name in source_names:
                        stats = source_stats[source_name]
                        to_sync[f"sources/{source_name}/loss_sum"] = stats["loss_sum"].item()
                        to_sync[f"sources/{source_name}/token_count"] = stats["token_count"].item()
                        to_sync[f"sources/{source_name}/corrects"] = stats["corrects"].item()

                synced_metrics = dist_mean_dict(to_sync)
                synced_token_count = max(float(synced_metrics["token_count/out"]), 1e-8)
                synced_metrics["accuracy/out"] = float(synced_metrics["corrects/out"]) / synced_token_count

                # BPB: all_reduce sum nats and bytes across ranks, then divide
                _bpb_nats = bpb_nats_sum.clone()
                _bpb_bytes = bpb_bytes_sum.float().clone()
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(_bpb_nats, op=dist.ReduceOp.SUM)
                    dist.all_reduce(_bpb_bytes, op=dist.ReduceOp.SUM)
                total_bytes = float(_bpb_bytes.item())
                synced_metrics["bpb"] = float(_bpb_nats.item()) / (math.log(2) * total_bytes) if total_bytes > 0 else float("nan")

                if args.track_source_metrics:
                    for source_name in source_names:
                        source_tokens = max(float(synced_metrics[f"sources/{source_name}/token_count"]), 1e-8)
                        synced_metrics[f"sources/{source_name}/loss"] = (
                            float(synced_metrics[f"sources/{source_name}/loss_sum"]) / source_tokens
                        )
                        synced_metrics.pop(f"sources/{source_name}/loss_sum")
                        synced_metrics[f"sources/{source_name}/accuracy"] = (
                            float(synced_metrics[f"sources/{source_name}/corrects"]) / source_tokens
                        )
                        synced_metrics.pop(f"sources/{source_name}/corrects")
                metrics.update(synced_metrics)

                if get_is_master():
                    metric_logger.log(metrics)

                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  bpb: {metrics['bpb']:.4f}"
                    f"  accuracy: {metrics['accuracy/out']:>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                )

            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainAnswerOnlyArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
