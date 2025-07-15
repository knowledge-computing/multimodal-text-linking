cd LIGHT

python train.py --config configs/light.yaml

python inference.py --test_dataset MapText_test --out_file predict.json --model_dir ./_runs/layoutLMv4_finetune_MapText_poly_only__v0 --anno_path /home/yaoyi/shared/spotter-data/icdar-maptext/competition-data/icdar24-test-png-annotations.json --img_dir /home/yaoyi/shared/spotter-data/icdar-maptext/competition-data/icdar24-test-png/test_images/

