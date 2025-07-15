cd LIGHT

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=14476 pretrain_poly.py --config configs/pretrain_poly.yaml