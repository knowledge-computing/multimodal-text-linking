#!/bin/bash -l 
#SBATCH --time=18:00:00 
#SBATCH --ntasks=8 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=lin00786@umn.edu 
#SBATCH -p msigpu 
#SBATCH --gres=gpu:1 

module load python3 
module load gcc/13.1.0-mptekim 
source activate map 

cd /users/8/lin00786/work/MapText/text_linking/ 

python pretrain_poly.py --config configs/pretrain_poly.yaml