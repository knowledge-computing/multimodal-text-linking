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
  --anno_path icdar24-test-png-annotations.json \
  --img_dir icdar24-test-png/test_images/
```

## 📁 Notes

- Create a Conda environment from [env.yaml](env.yaml)
- All dataset configurations are in the `dataset/buildin.py`. You need to update the paths to your dataset and annotations. Contact [Yijun Lin](https://linyijun.github.io/) if you want to use the pretraining datasets. We use [ICDAR24 MapText competition Rumsey dataset](https://rrc.cvc.uab.es/?ch=28) for finetuning and testing.
- All config files are in the `configs` directory. You can modify hyperparameters or dataset settings.
- You can download model weights from Google Drive: [Polygon Pretrain Weights](https://drive.google.com/drive/folders/1Qo0u1cVdrQ3vQOBH_PUGNF7BOjDbG3OP?usp=drive_link), [LIGHT Pretrain Weights](https://drive.google.com/drive/folders/1YhqYR7qjL0lp-gCnv0BYxin2FfvRdupD?usp=drive_link), [LIGHT Fintuned Weights](https://drive.google.com/drive/folders/16Ups2gbW7EVAttD17KPTF3V5O-_Zd96m?usp=drive_link)

## 🔗 References

If you find this repository useful in your own work, we would appreciate a citation to the accompanying paper:

```bibtex
@inproceedings{weinman2024counting,
   authors = {Lin, Yijun and Olson, Rhett and Wu, Junhan and Chiang, Yao-Yi and Weinman, Jerod},
   title = {LIGHT: Multi-Modal Text Linking on Historical Maps},
   booktitle = {19th International Conference on Document Analysis and Recognition ({ICDAR} 2025)},
   series = {Lecture Notes in Computer Science},
   publisher = {Springer},
   location = {Wuhan, China},
   year = {2025}
}
```

