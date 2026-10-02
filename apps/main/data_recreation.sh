#!/bin/bash
#SBATCH --job-name=comma
#SBATCH --time=08:00:00
#SBATCH --output=/project/aip-craffel/gsa/.slurm/%j.out
#SBATCH --nodes=2                       
#SBATCH --cpus-per-task=8            
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=l40s:4
#SBATCH --ntasks-per-node=1             
###SBATCH --exclude "rack12-05,rack12-06,rack01-08,rack04-10,rack06-04,rack07-04"
###SBATCH --mem=0
###SBATCH --gres=gpu:l40s:4
###SBATCH --output=/scratch/gsa/.slurm/%j.out

# sbatch apps/main/data_recreation.sh
# bash apps/main/data_recreation.sh

PROJECT_FOLDER=$HOME
# --- Environment Setup (Adjust for your environment) ---
# module load StdEnv/2023   openmpi/4.1.5 cuda/12.2 python/3.10.13
# cleanup
module --force purge
module load   StdEnv/2023 gcc/12.3                     
module load cuda/12.2                    
module load nccl/2.18.3                  
module load python/3.10.13               
cd $PROJECT_FOLDER/lingua;
source .venv/bin/activate
# --------------------------------------------------------

# --- Multi-Node Rendezvous Setup ---
# 1. Get the primary (master) node's hostname
MASTER_HOST=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# 2. Resolve hostname to IP address (crucial for C10D backend)
MASTER_ADDR=$(getent hosts $MASTER_HOST | awk '{ print $1 }' | head -n 1)
MASTER_PORT=29501
RDZV_ID=$SLURM_JOB_ID # Use the unique Slurm job ID for coordination
NPROCS=$(( ${SLURM_NNODES:-1} * ${SLURM_NTASKS_PER_NODE:-4} ))
if [ -z "${RDZV_ID}" ]; then RDZV_ID=101; fi
if [ -z "${NPROCS}" ]; then NPROCS=4; fi

echo "Master Node: $MASTER_HOST ($MASTER_ADDR)"
echo "Rendezvous Port: $MASTER_PORT"
echo "Job ID: $RDZV_ID"


# --- Execute torchrun across all allocated nodes ---
# We use srun to launch the same torchrun command on all nodes simultaneously.

MODEL_NAME="aya"
MODEL_NAME="qwen3"
MODEL_NAME="mbert"
# MODEL_NAME="byt5"
MODEL_NAME="tokenmonster"
MODEL_NAME="gpt4o"
MODEL_NAME="tekken"
MODEL_NAME="xglm"
MODEL_NAME="bloom"
MODEL_NAME="gemma2"
MODEL_NAME="gpt2"
MODEL_NAME="phi-3"
MODEL_NAME="comma"
MODEL_NAME="llama_1B"

CONFIG_FILE="apps/main/data_configs_trillium/toksuit_${MODEL_NAME}.yaml"
CONFIG_FILE="apps/main/data_configs/toksuit_${MODEL_NAME}.yaml"

export REPORT_BYTES="True"
export DUMP_DOCS="False"
export PRINT_DOCS="False"

if [ "${MODEL_NAME}" == "gpt4o" ]; then
	export TIKTOKEN_CACHE_DIR="$SCRATCH/.cache/tiktoken"
	echo "Running gpt-4o tiktoken test"
	python -c "import tiktoken;model_path='gpt-4o';tiktoken.encoding_for_model(model_path)"
fi

echo gpus on node $SLURM_GPUS_ON_NODE  nodes $SLURM_NNODES rdzv_id $RDZV_ID

srun  --ntasks-per-node=1 -N $SLURM_NNODES \
torchrun \
    --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m apps.main.recreate_training_data \
    config=$CONFIG_FILE 
	# > $MODEL_NAME-data-recreation.log 2>&1


