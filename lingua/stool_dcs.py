# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, MutableSequence

from omegaconf import OmegaConf

from lingua.args import dataclass_from_dict


@dataclass
class StoolArgs:
    config: Any = None
    launcher: str = "sbatch"  # Can be sbatch or bash if already in salloc
    script: str = "apps.main.train"  # The script to run.
    copy_code: bool = True  # Wether to copy code to dump dir
    dirs_exists_ok: bool = (
        True  # Wether to copy new code and config and run regardless that dir exists
    )
    override: bool = False  # Wether to delete dump dir and restart
    nodes: int = 1  # The number of nodes to run the job on.
    ngpu: int = 8  # The number of GPUs required per node.
    ncpu: int = 16  # The number of CPUs allocated per task.
    mem: str = ""  # The amount of memory to allocate.
    anaconda: str = "default"  # The path to the anaconda environment.
    venv: str = "/h/371/gsaltintas/lingua/.venv" # The path to the virtual environment (alternative to anaconda).
    constraint: str = ""  # The constraint on the nodes.
    exclude: str = ""  # The nodes to exclude.
    time: int = 1200  # The time limit of the job (in minutes).
    account: str = ""
    qos: str = ""
    partition: str = ""
    stdout: bool = False
    priority: str = "normal"
    # data_dir: str = "/fsx/craffel/common-pile-chunked/"
    data_dir: str = "/mfs1/u/gsaltintas/data/flexitok/" 
    host: str = "killarney"  # The host to determine module load commands, e.g. killarney or hopper
    gpu_type: str = ""  # The type of GPU to request in sbatch command, e.g. l40s or h100

SBATCH_COMMAND = """#!/bin/bash

{exclude}
{qos}
{account}
{constraint}
#SBATCH --job-name={name}
#SBATCH --nodes={nodes}
{sbatch_option}
####SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task={ncpu}
#SBATCH --time={time}
###SBATCH --output={dump_dir}/logs/%j.stdout
###SBATCH --error={dump_dir}/logs/%j.stderr

#SBATCH --begin=now+0minutes
###SBATCH --mail-type=ALL
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=gulsena.altintas@mail.utoronto.ca
#SBATCH --requeue

#SBATCH --open-mode=append
{module_load_command}
echo "Modules loaded"
echo $(module list)

echo Job output will be written to {dump_dir}/logs/%j.stdout and {dump_dir}/logs/%j.stderr
# Mimic the effect of "conda init", which doesn't work for scripts
{activate_command}

{go_to_code_dir}

# export OMP_NUM_THREADS=1 
export LAUNCH_WITH="SBATCH"
export DUMP_DIR={dump_dir}
# export TMPDIR=/mfs1/u/gsaltintas/tmp

#{copy_data_command}

if [ -z "$SLURM_NTASKS" ]; then
    export SLURM_NTASKS={tasks}
fi

echo total tasks $SLURM_NTASKS
echo $(which python)

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_SOCKET_IFNAME="eth0" # or "ib0" for InfiniBand
export NCCL_DEBUG="INFO"
export NCCL_DEBUG_SUBSYS="ALL"
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"

# otherwise init times out on killarney
# https://discuss.pytorch.org/t/torch-distributed-init-process-group-hangs-with-4-gpus-with-backend-nccl-but-not-gloo/149061/6
# export NCCL_P2P_DISABLE=1


NODES=$(python - "$SLURM_JOB_NODELIST" <<'PYEOF'
import re, sys
nodelist = sys.argv[1]
m = re.match(r"^([^\[]+)(?:\[([^\]]+)\])?$", nodelist)
prefix, ranges = m.group(1), m.group(2)
out = []

if ranges is None:
    out.append(prefix)
else:
    for part in ranges.split(","):
        if "-" in part:
            a, b = part.split("-")
            width = len(a)
            for n in range(int(a), int(b) + 1):
                out.append(prefix + str(n).zfill(width))
        else:
            out.append(prefix + part)

print("\\n".join(out))
PYEOF
)
echo "Resolved nodes: $NODES"


# There is no srun (and no scontrol) on this cluster, so Slurm never puts a
# process on each node/GPU for us, and we can't ask scontrol to expand
# $SLURM_JOB_NODELIST's bracket notation (e.g. "quartet[2,5]"). torchrun
# itself only manages local worker spawning + rendezvous on whichever node
# it runs on - it doesn't start processes on other machines. So: expand the
# node list ourselves, write a wrapper script every node will run, and open
# one tmux window per node that ssh's in and starts it there with the right
# --node_rank. `tmux attach -t $SESSION` (from the node running this batch
# script) shows every node's live output.
export MASTER_ADDR=$(hostname -I | awk '{{print $1}}')
FIRST_NODE=$(echo "$NODES" | head -n1)
export MASTER_ADDR=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$FIRST_NODE" "hostname -I | awk '{{print \$1}}'")
export MASTER_PORT=$(( 20000 + RANDOM % 40000 ))

echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo outputs being logged at {log_output}


echo $DUMP_DIR/run_node.sh
cat > $DUMP_DIR/run_node.sh <<NODEEOF
#!/bin/bash
set -e
echo "Running on node \$SLURM_NODEID, rank \$1 \$(hostname)"
{go_to_code_dir}
{activate_command}
export OMP_NUM_THREADS=1
export DUMP_DIR=$DUMP_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export USE_NCCL=1
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME="eth0" # or "ib0" for InfiniBand
export NCCL_SOCKET_IFNAME="eno16npo"
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT

torchrun \\
  --nnodes=$SLURM_NNODES \\
  --nproc_per_node={ngpus} \\
  --node_rank=\$1 \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  -m {script} config=$DUMP_DIR/base_config.yaml

#   --rdzv_id=$SLURM_JOB_ID \\
#   --rdzv_backend=c10d \\
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
#   --rdzv_conf join_timeout=1200,close_timeout=1200,timeout=1200 \\

NODEEOF
chmod +x $DUMP_DIR/run_node.sh

mkdir -p $DUMP_DIR/logs

SESSION=train_$SLURM_JOB_ID
NODE_RANK=0
for NODE in $NODES; do
    WINDOW=node-$NODE_RANK
    LOG=$DUMP_DIR/logs/$SLURM_JOB_ID-$NODE_RANK.log
    CMD="ssh -v -o BatchMode=yes -o StrictHostKeyChecking=no $NODE 'bash $DUMP_DIR/run_node.sh $NODE_RANK' 2>&1 | tee $LOG; tmux wait-for -S $WINDOW-done"
    echo $NODE $CMD
    if [ "$NODE_RANK" -eq 0 ]; then
        tmux new-session -d -s $SESSION -n $WINDOW 
        tmux set-option -t $SESSION remain-on-exit on
        tmux new-window -t $SESSION -n $WINDOW "$CMD"
    else
        tmux new-window -t $SESSION -n $WINDOW "$CMD"
    fi
    NODE_RANK=$((NODE_RANK + 1))
done

echo "tmux session '$SESSION' started - attach with: tmux attach -t $SESSION"
echo "per-node ssh/torchrun logs also written to $DUMP_DIR/logs/$SLURM_JOB_ID-*.log"

NODE_RANK=0
for NODE in $NODES; do
    tmux wait-for node-$NODE_RANK-done
    NODE_RANK=$((NODE_RANK + 1))
done

# tmux kill-session -t $SESSION 2>/dev/null || true
"""


def copy_dir(input_dir: str, output_dir: str) -> None:
    print(f"Copying : {input_dir}\n" f"to      : {output_dir} ...")
    assert os.path.isdir(input_dir), f"{input_dir} is not a directory"
    assert os.path.isdir(output_dir), f"{output_dir} is not a directory"
    rsync_cmd = (
        f"rsync -arm --copy-links "
        f"--exclude .venv "
        f"--include '**/' "
        f"--include '*.py' "
        f"--exclude='*' "
        f"{input_dir}/ {output_dir}"
    )
    print(f"Copying command: {rsync_cmd}")
    subprocess.call([rsync_cmd], shell=True)
    print("Copy done.")


def retrieve_max_time_per_partition() -> Dict[str, int]:
    # retrieve partition max times (a bit slow)

    sinfo = json.loads(subprocess.check_output("sinfo --json", shell=True))["sinfo"]
    max_times: Dict[str, int] = {}

    for info in sinfo:
        if info["partition"]["maximums"]["time"]["infinite"]:
            max_times[info["partition"]["name"]] = 14 * 24 * 60  # 14 days
        else:
            max_times[info["partition"]["name"]] = info["partition"]["maximums"][
                "time"
            ][
                "number"
            ]  # in minutes

    return max_times


def validate_args(args) -> None:
    # Set maximum time limit if not specified
    if args.time == -1:
        max_times = retrieve_max_time_per_partition()
        args.time = max_times.get(
            args.partition, 3 * 24 * 60
        )  # Default to 3 days if not found
        print(
            f"No time limit specified, using max time for partitions: {args.time} minutes"
        )

    if args.constraint:
        args.constraint = f"#SBATCH --constraint={args.constraint}"

    if args.account:
        args.account = f"#SBATCH  --account={args.account}"

    if args.qos:
        args.qos = f"#SBATCH --qos={args.qos}"

    if getattr(args, "exclude", ""):
        args.exclude = f"#SBATCH --exclude={args.exclude}"

    if hasattr(args, "venv") and args.venv:
        if not args.venv.endswith("/bin/activate"):
            args.venv = f"{args.venv}/bin/activate"
        assert os.path.isfile(args.venv), f"Virtual environment not found at {args.venv}"
        args.anaconda = ""  # Ensure anaconda is not used if venv is specified

    if hasattr(args, "anaconda") and args.anaconda:
        if args.anaconda == "default":
            args.anaconda = (
                subprocess.check_output("which python", shell=True)
                .decode("ascii")
                .strip()
            )
        else:
            args.anaconda = f"{args.anaconda}/bin/python"
        assert os.path.isfile(args.anaconda)

    args.mem = args.mem or "0"

    # assert args.partition
    assert args.ngpu > 0
    assert args.ncpu > 0
    assert args.nodes > 0
    assert args.time > 0

def modify_for_ccdb(args: StoolArgs):
    args.config["dump_dir"] = args.config["dump_dir"].replace("/fsx/craffel/toksuite/lingua_logs", "/mfs1/u/gsaltintas/train")
    args.config["dump_dir"] = args.config["dump_dir"].replace("/fsx/craffel/lingua_logs", "/mfs1/u/gsaltintas/train")
    # import code; code.interact(local=dict(globals(), **locals()))
    if args.config.get("data") is not None:
        data_conf = args.config["data"]
        print(args.config["data"].get("tokenizer") is not None and args.config["data"]["tokenizer"].get("path") is not None)
        if args.config["data"].get("tokenizer") is not None and args.config["data"]["tokenizer"].get("path") is not None:
            tok_args = deepcopy(args.config["data"]["tokenizer"])
            tok_args.update({"path": tok_args["path"].replace("/fsx/craffel/lingua_logs", "/mfs1/u/gsaltintas/train")})
            print(tok_args)
            data_conf["tokenizer"] = tok_args
        if data_conf.get("root_dir") is not None:
            # toksuite
            data_conf["root_dir"] = args.config["data"]["root_dir"].replace("/scratch/craffel/lingua/data/tokenizer_training", "/mfs1/u/gsaltintas/data/tokenizer_training")
            data_conf["root_dir"] = args.config["data"]["root_dir"].replace("/scratch/craffel/lingua/data", "/mfs1/u/gsaltintas/data")
        args.config["data"] = data_conf 
    if args.config.get("ckpt_dir", None) is not None:
        args.config["ckpt_dir"] = args.config["ckpt_dir"].replace("/fsx/craffel/lingua_logs", "/mfs1/u/gsaltintas/train")
    if args.config.get("checkpoint", None) is not None:
        args.config["checkpoint"]["path"] = args.config["checkpoint"].get("path", "").replace("/fsx/craffel/toksuite/lingua_logs", "/mfs1/u/gsaltintas/train")
        args.config["checkpoint"]["path"] = args.config["checkpoint"].get("path", "").replace("/fsx/craffel/lingua_logs", "/mfs1/u/gsaltintas/train")
        if args.config["checkpoint"].get("init_ckpt_path", None) is not None:
            args.config["checkpoint"]["init_ckpt_path"] = args.config["checkpoint"].get("init_ckpt_path", "").replace("/fsx/craffel/toksuite/lingua_logs", "/mfs1/u/gsaltintas/train")
            args.config["checkpoint"]["init_ckpt_path"] = args.config["checkpoint"].get("init_ckpt_path", "").replace("/fsx/craffel/lingua_logs", "/mfs1/u/gsaltintas/train")
    print(args.config["dump_dir"])
    # print(args.config["data"].get("tokenizer"))


def launch_job(args: StoolArgs):
    # Set up args default and validate them depending on the cluster or partition requested
    modify_for_ccdb(args)
    validate_args(args)
    dump_dir = args.config["dump_dir"]
    job_name = args.config["name"]

    if "data" in args.config:
        data_dir = args.data_dir
        data_root_dir = args.config["data"].get("root_dir", "")
        if data_dir.startswith("s3://"):
            copy_data_command = f"srun --ntasks-per-node=1 s5cmd cp '{data_dir.removesuffix('/')}/*' {data_root_dir}/"
        else:
            copy_data_command = f"srun --ntasks-per-node=1 bash -c 'mkdir -p {data_root_dir} && rsync -arm {data_dir} {data_root_dir} --exclude .venv --exclude __pycache__ '"
    else:
        copy_data_command = ""

    print("Creating directories...")
    # import code; code.interact(local=dict(globals(), **locals()))
    os.makedirs(dump_dir, exist_ok=args.dirs_exists_ok or args.override)
    if args.override:
        confirm = input(
            f"Are you sure you want to delete the directory '{dump_dir}'? This action cannot be undone. (yes/no): "
        )
        if confirm.lower() == "yes":
            shutil.rmtree(dump_dir)
            print(f"Directory '{dump_dir}' has been deleted.")
        else:
            print("Operation cancelled.")
            return
    os.makedirs(os.path.join(dump_dir, "logs"), exist_ok=args.dirs_exists_ok)
    if args.copy_code:
        os.makedirs(f"{dump_dir}/code", exist_ok=args.dirs_exists_ok)
        print("Copying code ...")
        copy_dir(os.getcwd(), f"{dump_dir}/code")

    print("Saving config file ...")
    with open(f"{dump_dir}/base_config.yaml", "w") as cfg:
        cfg.write(OmegaConf.to_yaml(args.config))

    if args.anaconda:
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        conda_env_path = os.path.dirname(os.path.dirname(args.anaconda))
        activate_command = f"eval \"$({conda_exe} shell.bash hook)\" && source activate {conda_env_path}"
    elif args.venv:
        conda_exe = "source"
        conda_env_path = args.venv
        activate_command = f"source {args.venv}"
    else:
        conda_exe = ""
        conda_env_path = ""
        activate_command = ""


    if args.gpu_type:
        gres_str = f"--gres=gpu:{args.gpu_type}:{args.ngpu}"
    else:
        gres_str = f"--gres=gpu:{args.ngpu}"
    module_load_command = "module --force purge\n\n"
    if hasattr(args, "host") and args.host == "killarney":
        module_load_command += "module load slurm/killarney/25.05.6\n"
    module_load_command += """
module load   StdEnv/2023  gcc/12.3  openmpi/4.1.5
# module load cuda/12.2                
# module load nccl/2.18.3                  
# module load python/3.10.13        
# module load mii/1.1.2 ucx/1.14.1   
#             
module load cuda/13.2                   
module load nccl/2.29.7                 
module load python/3.10.13        
module load mii/1.1.2 ucx/1.14.1
"""
    log_output = (
        "-o $DUMP_DIR/logs/%j_%t.out -e $DUMP_DIR/logs/%j_%t.err"
        if not args.stdout
        else ""
    )
    sbatch_option = f"""
#SBATCH --ntasks-per-gpu=1          # crucial - only 1 task per dist per node!
#SBATCH {gres_str}
#SBATCH --mem={args.mem}
#SBATCH --output=/mfs1/u/gsaltintas/.slurm/%j.out
"""

    sbatch = SBATCH_COMMAND.format(
        name=job_name,
        script=args.script,
        dump_dir=dump_dir,
        nodes=args.nodes,
        tasks=args.nodes * args.ngpu,
        nodes_per_run=args.nodes,
        ngpus=args.ngpu,
        ncpu=args.ncpu,
        mem=args.mem,
        qos=args.qos,
        account=args.account,
        constraint=args.constraint,
        exclude=args.exclude,
        time=args.time,
        partition=args.partition,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        log_output=log_output,
        go_to_code_dir=f"cd {dump_dir}/code/" if args.copy_code else "",
        priority=args.priority,
        copy_data_command=copy_data_command,
        activate_command=activate_command,
        module_load_command=module_load_command,
        gpu_type=args.gpu_type,
        sbatch_option=sbatch_option,
        gres_str=gres_str
    )

    print("Writing sbatch command ...")
    with open(f"{dump_dir}/submit.slurm", "w") as f:
        f.write(sbatch)

    print(f"Submitting job with {args.launcher}...")
    os.system(f"{args.launcher} {dump_dir}/submit.slurm")

    print("Done.")


if __name__ == "__main__":
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        mode: LMTransformerArgs

    @dataclass
    class LMTransformerArgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgs
    or just name=tictac for top level attributes.
    """
    # import code; code.interact(local=locals()|globals() )
    args = OmegaConf.from_cli()

    if isinstance(args.config, MutableSequence):
        args.config = OmegaConf.merge(*[OmegaConf.load(c) for c in args.config])
    else:
        args.config = OmegaConf.load(args.config)
    args = dataclass_from_dict(StoolArgs, args)
    launch_job(args)
