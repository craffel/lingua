# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os

report_bytes=os.environ.get("REPORT_BYTES", "False") == "True"
dump_data=os.environ.get("DUMP_DATA", "False") == "True"
dump_docs=os.environ.get("DUMP_DOCS", "False") == "True"
import gc
import json
import logging
import sys
import time
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import wandb
import xformers.profiler
from omegaconf import OmegaConf
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import lr_scheduler

from apps.main.transformer import (
    LMTransformer,
    LMTransformerArgs,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token,
    tp_parallelize,
)
from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    check_model_value_range,
    clean_env,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_master_port,
    get_world_size,
    init_signal_handler,
    parallelize_model,
    requeue_slurm_job,
    setup_env,
    setup_torch_distributed,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer
from lingua.probe import AutoProbeD
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.stool import StoolArgs, launch_job
from lingua.tokenizer import build_tokenizer

logger = logging.getLogger()
from apps.main.train import (
    TrainArgs,
    TrainState,
    every_n_steps,
    set_preemption_flag,
    validate_train_args,
)

preemption_flag = dict(flag=False)


class FakeDeviceMesh:
    """Mock device mesh to simulate distributed behavior"""
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.ndim = 1  # Two dimensions: dp_replicate 
        self.mesh_dim_names = ["dp_replicate"]
    
    def __getitem__(self, key):
        if key == "dp_replicate":
            return FakeSubMesh(self.rank, self.world_size)
        elif key == "dp_shard":
            return FakeSubMesh(0, 1)  # Assuming no sharding for simplicity
        return self
    
    def get_local_rank(self, key=None):
        if key == "dp_replicate":
            return self.rank
        elif key == "dp_shard":
            return 0  # Assuming no sharding for simplicity
        return self.rank
    
    def size(self):
        return self.world_size

class FakeSubMesh:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size
        # simulate 1 node?
        self.ndim = 1
    
    def get_local_rank(self):
        return self._rank
    
    def size(self):
        return self._size

def setup_fake_distributed(rank, world_size):
    """Setup fake distributed environment for single-process simulation"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(get_master_port(os.environ.get("SLURM_JOB_ID")))
    import torch.distributed as dist

    dist.init_process_group(
        backend='nccl',
        # backend='gloo',
        init_method="env://",
        rank=0,
        world_size=1
        # world_size=world_size
    )
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # Don't actually initialize distributed - just set environment
    return FakeDeviceMesh(rank, world_size)

def recreate_rank_data(args: TrainArgs, rank: int, world_size: int):
    with ExitStack() as context_stack:
        torch.backends.cuda.enable_cudnn_sdp(False)
        TOTAL_BYTES = 0
        # tokenizer - shared across all ranks
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(
            args,
            tokenizer.n_words,
        )

        if get_is_master():
            import os
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        logger.info(f"Starting job: {args.name}")
        if rank is None:
            print("Distributeeed", args.distributed)
            setup_torch_distributed(args.distributed)
            # import os
            # local_rank = int(os.environ.get("LOCAL_RANK", 0))
            # torch.cuda.set_device(local_rank)
            world_mesh = get_device_mesh(args.distributed)
            logger.info(f"Setup torch distributed.")
        else:
            # Setup fake distributed environment
            world_mesh = setup_fake_distributed(rank, world_size)
            logger.info(f"Setup fake distributed mesh.")
        # Calculate dp_rank and dp_degree exactly like in training
        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()

        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()
        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        ## will hold our data
        training_data = []
        written_lines = 0
        curr_sub_file=0
        dump_file = Path(args.dump_dir) / f"train_data_{dp_rank}.{curr_sub_file:02d}-{dp_degree}.jsonl"
        logger.info(f"Dumping training data to {dump_file}")

        torch.manual_seed(args.seed)
        logger.info("Building model")
        
        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        ## TODO: not sure if we need to modify
        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        if rank is None:
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

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)

        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            # if args.checkpoint.load_init_optimizer_state:
                # load_from_checkpoint(args.checkpoint.init_ckpt_path, model, optimizer, model_key="model") # Put model_key="" if its directly the model checkpoint
            # else:
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
            model.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        if args.checkpoint.save_init_ckpt:
            _ = checkpoint.save(
                model,
                optimizer,
                train_state,
                args,
                device_mesh=world_mesh,
            )

        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            # torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(
                args.data,
                state=train_state.data_loader_state,
            )
        )
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        saved = False
        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get batch
            # curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            if not dump_data and report_bytes:
                total_bytes = 0
                examples = batch[:, :, 0]
                for example in examples:
                    text = tokenizer.decode(example.tolist(), skip_special_tokens=False)
                    nbytes = len(text.encode('utf-8'))
                    total_bytes += nbytes
                    TOTAL_BYTES += nbytes
            logger.info(f"Rank {dp_rank} - Step {train_state.step} acc {train_state.acc_step} -  Total bytes (GB): {TOTAL_BYTES/1e9}")
            if dump_data:
                batch = torch.tensor(
                    batch,
                    dtype=torch.long,
                )
            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()
            if dump_data:
                input_ids = batch[:, :, 0].cuda()
                labels = batch[:, :, 1].cuda()
                data_load_time = round(timer() - data_load_start, 4)
                nwords_since_last_log += input_ids.numel()
                bsz, seqlen = labels.shape

                for i, example in enumerate(input_ids):
                    # Decode input ids to text
                    training_data_txt = tokenizer.decode(example.tolist(), skip_special_tokens=False)
                    num_bytes = len(training_data_txt.encode('utf-8'))
                    training_data.append({"step": train_state.step, "grad_acc_step": train_state.acc_step, "seq_idx": i, "text": training_data_txt, "num_bytes": num_bytes})

            if dump_data and every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0):
                logger.info("Dumping training data")
                with open(dump_file, "a") as f:
                    for item in training_data:
                        f.write(json.dumps(item) + "\n")
                        written_lines += 1 
                training_data = []
                if written_lines >= 800000:
                    curr_sub_file += 1
                    dump_file = Path(args.dump_dir) / f"train_data_{dp_rank}.{curr_sub_file:02d}-{dp_degree}.jsonl"
                    written_lines = 0

            # forward
            # start_timer = torch.cuda.Event(enable_timing=True)
            # end_timer = torch.cuda.Event(enable_timing=True)
            # start_timer.record()

            ## deleted probes

            # loss = model(input_ids, labels)

            # # We scale loss with grad_acc_steps so the gradient is the same
            # # regardless of grad_acc_steps
            # loss = loss / args.grad_acc_steps
            # # backward on scaled loss to create scaled gradients
            # loss.backward()
            # # For logging we undo that scaling
            # loss = loss.detach() * args.grad_acc_steps

            # # optimizer step
            # grad_norm = -1.0
            if train_state.acc_step == 0:
                # grad_norm = torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), max_norm=args.optim.clip, foreach=True
                # )

                # grad_norm = (
                #     grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                # ).item()

                # optimizer.step()
                # scheduler.step()
                # optimizer.zero_grad()
                train_state.step += 1
                # if not dump_data:
                #     print(f"Step {train_state.step}")
            if dump_data:
                import os
                print(f"Rank {os.environ.get('RANK')} - Completed step {train_state.step} acc step {train_state.acc_step}")
            # updates the scale for next iteration
            # training iteration complete
            # end_timer.record()

            # torch.cuda.synchronize()

            # curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                xformers.profiler.step()

            # log metrics
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

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
                # FLOPS = (
                #     get_num_flop_per_token(
                #         model_param_count - args.model.vocab_size * args.model.dim,
                #         args.model.n_layers,
                #         args.model.dim,
                #         args.data.seq_len,
                #     )
                #     * wps
                # )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            # "FLOPS": FLOPS,
                            # "curr_iter_time": curr_iter_time,
                            # "data_load_time": data_load_time,
                        },
                        "optim": {
                            # "grad_norm": grad_norm,
                            # "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                # to_sync["loss/out"] = loss.item()
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    # f"  loss: {round(loss.item(),4):>7}"
                    # f"  grad: {grad_norm:.2e}"
                    # f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    # f"  iter: {curr_iter_time:>7}"
                    # f"  data: {data_load_time:>5}"
                    # f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )
                import os
                print(f"Rank {os.environ.get('RANK')} - Step {train_state.step} acc {train_state.acc_step} - total_tokens: {total_tokens}")

            saved = False
            # if every_n_steps(
            #     train_state, args.checkpoint.dump.every, acc_step=0
            # ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
            #     saved = checkpoint.save(
            #         model,
            #         optimizer,
            #         train_state,
            #         args,
            #         device_mesh=world_mesh,
            #     )
            ## deleted eval loops
            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    with open(dump_file, "a") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
    # if not saved:
    #     checkpoint.save(
    #         model,
    #         optimizer,
    #         train_state,
    #         args,
    #         device_mesh=world_mesh,
    #     )
    gc.collect()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """

    ## either call with torchrun regularly e.g. and it will write all files in sync
    # torchrun --nproc-per-node 8  --nnodes=1 -m apps.main.recreate_training_data config=apps/main/data_configs/toksuit_gpt4o_torchrun.yaml
    # torchrun --nproc-per-node 4  --nnodes=2 -m apps.main.recreate_training_data config=apps/main/data_configs_trillium/toksuit_gemma2.yaml
    ## Later upload to hf
    ## hf upload-file --path $SCRATCH/data_recreation-2/gemma-2b/* --repo-id r-three/training_data_detokenized-gemma-2b-torchrun --repo-type dataset --token $HF_TOKEN
    ## or run from a single gpu with
    # python -m apps.main.recreate_training_data config=apps/main/data_configs/toksuit_$tok.yaml rank=$rank world_size=8
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # Add specific arguments -- don't call these with torchrun
    rank = cli_args.get('rank', None)
    world_size = cli_args.get('world_size', 8)

    # Remove custom args from cli_args
    if 'rank' in cli_args:
        del cli_args.rank
    if 'world_size' in cli_args:
        del cli_args.world_size
    
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config
    print(f"Variables: rank: {rank}, world_size: {world_size}")

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    recreate_rank_data(cfg, rank, world_size)


if __name__ == "__main__":
    main()
