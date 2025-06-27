#!/bin/bash -l 
#SBATCH --time=6:00:00 
#SBATCH -N 1
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=30GB 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=lin00786@umn.edu 
#SBATCH -p yaoyi 
#SBATCH --gres=gpu:a100:1 

module load python3 
module load gcc/13.1.0-mptekim 
source activate map 

cd /users/8/lin00786/work/MapText/text_linking/ 

# python inference1.py --test_dataset HierText_test --output_file predict.json --model_dir ./_runs/poly_only__v0

python inference1.py --test_dataset test --out_file lithium.json --model_dir _runs/best/ --anno_path /home/yaoyi/shared/critical-maas/12month-text-extraction/spot/lithium.json  --img_dir /home/yaoyi/shared/critical-maas/12month-text-extraction/img_crops/lithium

python inference1.py --test_dataset test --out_file nickel.json --model_dir _runs/best/ --anno_path /home/yaoyi/shared/critical-maas/12month-text-extraction/spot/nickel.json  --img_dir /home/yaoyi/shared/critical-maas/12month-text-extraction/img_crops/nickel

python inference1.py --test_dataset test --out_file regionalporcu.json --model_dir _runs/best/ --anno_path /home/yaoyi/shared/critical-maas/12month-text-extraction/spot/regionalporcu.json  --img_dir /home/yaoyi/shared/critical-maas/12month-text-extraction/img_crops/regionalporcu

python inference1.py --test_dataset test --out_file zinc.json --model_dir _runs/best/ --anno_path /home/yaoyi/shared/critical-maas/12month-text-extraction/spot/zinc.json  --img_dir /home/yaoyi/shared/critical-maas/12month-text-extraction/img_crops/zinc

python inference1.py --test_dataset MapText_test --out_file predict_CREPE_125336.json --model_dir _runs/best/ --anno_path data/MapText_CREPE_125336.json --img_dir /home/yaoyi/shared/spotter-data/icdar-maptext/competition-data/icdar24-test-png/test_images/