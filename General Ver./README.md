Compressed Convolution Networks for Graphs (CoCN)
---

This is the official implementation of General CoCN for `Compressed Convolution Networks on Graphs`.

![CoCN Highlight](../highlight.png)


# Preparation
## Requirements

- Python 3.7+
- PyTorch 1.10.0+
- PyTorch Geometric 2.1.0+

To install requirements:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets are organized as:

```
.
├── data
│   └── webkb
│       └── cornell
│           ├── processed
│           │   ├── data.pt
│           │   ├── pre_filter.pt
│           │   └── pre_transform.pt
│           └── raw
│               ├── cornell_split_0.6_0.2_0.npz
│               ├── ...
│               ├── cornell_split_0.6_0.2_9.npz
│               ├── out1_graph_edges.txt
│               └── out1_node_feature_label.txt
```

# Getting Started

- Train with command line

```bash
python train.py  --cuda_num 'CUDA_VISIBLE_DEVICE' --nbatch 1 --testmode 'path/to/export/dir' --dataset 'CORNELL' --lr 1e-4 --epoch 500 --nTlayer 0 --nlayer 1 --nblock 3 --filter_size 5 --stride 5 --nh 10 --d_model 128 --dropout 0.5
python train_for_topk.py  --cuda_num 'CUDA_VISIBLE_DEVICE' --nbatch 1 --testmode 'path/to/export/dir' --dataset 'genius' --lr 1e-3 --epoch 500 --nTlayer 6 --nlayer 4 --nblock 1 --filter_size 5 --stride 5 --dropout 0.1 --base_size 3000 --nk 8
```

- Train with script

```bash
bash run.sh
```

- Monitor (TensorBoard required)

```bash
tensorboard --logdir='./export/path/to/export/dir' --port xxxx
```
