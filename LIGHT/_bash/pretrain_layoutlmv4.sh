#!/bin/bash -l 
#SBATCH --time=96:00:00 
#SBATCH -N 1
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=60GB 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=lin00786@umn.edu 
#SBATCH -p yaoyi 
#SBATCH --gres=gpu:a100:1 


module load python3 
module load gcc/13.1.0-mptekim 
source activate map 

cd /users/8/lin00786/work/MapText/text_linking/ 

# python pretrain.py --config configs/pretrain.yaml

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=14476 pretrain.py --config configs/pretrain_layoutlmv4.yaml