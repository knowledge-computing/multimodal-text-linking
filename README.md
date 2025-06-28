# LIGHT: Multi-Modal Text Linking for Historical Maps
This repository provides training and inference scripts for pretraining and fine-tuning the **LIGHT** model, which integrates polygon geometry and visual-textual features for text linking on historical maps.

## 📚 Pretraining

### 1. Polygon Encoder Pretraining

```bash
CUDA_VISIBLE_DEVICES="0" torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=14476 \
  pretrain_poly.py --config configs/pretrain_poly.yaml
```

### 2. Full LIGHT Model Pretraining

```bash
CUDA_VISIBLE_DEVICES="0" torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=14476 \
  pretrain.py --config configs/pretrain_light.yaml
```

## 🔧 Fine-Tuning

```bash
python train.py --config configs/light.yaml
```

## 🔍 Inference

```bash
python inference.py \
  --test_dataset MapText_test \
  --out_file predict.json \
  --model_dir ./_weights/finetune_light \
  --anno_path /home/yaoyi/shared/spotter-data/icdar-maptext/competition-data/icdar24-test-png-annotations.json \
  --img_dir /home/yaoyi/shared/spotter-data/icdar-maptext/competition-data/icdar24-test-png/test_images/
```

## 📁 Notes

- Update the paths to your dataset, annotations, and trained model checkpoints as needed.
- All the data configurations are in the dataset/buildin.py
- All configs can be modified under the `configs/` directory to adjust hyperparameters or dataset settings.
