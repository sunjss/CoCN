Compressed Convolution Networks for Graphs (CoCN)
---

This is the official implementation for Compressed Convolution Networks (CoCN), a general backbone for graph representation learning.

![CoCN Highlight](highlight.png)

## Getting Started
- CoCN is implemented with permutation generation and diagonal convolution.
- For the [Vanilla CoCN](https://proceedings.mlr.press/v202/sun23k.html) please refer to [Vanilla Ver.](https://github.com/sunjss/CoCN/blob/main/Vanilla%20Ver./README.md).
- We also seek to extend CoCN to encompass a broader range of graph tasks, please refer to [General Ver.](https://github.com/sunjss/CoCN/blob/main/General%20Ver./README.md) for our extension of CoCN. General CoCN enhances the vanilla version in several aspects:
  - Enabling support for graphs with implicit node features.
  - Enhancing scalability through sparsification and segmentation techniques.