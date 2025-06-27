#!/bin/bash -l 
#SBATCH --time=18:00:00 
#SBATCH -N 1
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=40GB 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=lin00786@umn.edu 
#SBATCH -p yaoyi 
#SBATCH --gres=gpu:a100:1 

module load python3 
module load gcc/13.1.0-mptekim 
source activate map 

cd /users/8/lin00786/work/MapText/text_linking/ 

python train.py --config configs/layoutlmv4.yaml

python inference.py --test_dataset MapText_test --out_file predict.json --model_dir ./_runs/layoutLMv4_finetune_MapText_poly_only__v0 --anno_path /home/yaoyi/shared/spotter-data/icdar-maptext/competition-data/icdar24-test-png-annotations.json --img_dir /home/yaoyi/shared/spotter-data/icdar-maptext/competition-data/icdar24-test-png/test_images/

