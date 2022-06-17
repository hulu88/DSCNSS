

# DSCNSS





# Deep subspace image clustering network with self-expression and self-supervision

- This is repository contains the code for the paper (Applied Intelligence, 2022)
- Chen, C., Lu, H*. Deep subspace image clustering network with self-expression and self-supervision. *Appl Intell* (2022). https://doi.org/10.1007/s10489-022-03654-6



# Abstract:

The subspace clustering algorithms for image datasets apply a self-expression coefficient matrix to obtain the correlation between samples and then perform clustering. However, such algorithms proposed in recent years do not use the cluster labels in the subspace to guide the deep network and do not get an end-to-end feature extraction and trainable clustering framework. In this paper, we propose a self-supervised subspace clustering model with a deep end-to-end structure, which is called Deep Subspace Image Clustering Network with Self-expression and Self-supervision (DSCNSS). The model embeds the self-supervised module into the subspace clustering. In network model training, alternating iterative optimization is applied to realize the mutual promotion of the self-supervised module and the subspace clustering module. Additionally, we design a new self-supervised loss function to improve the overall performance of the model further. To verify the performance of the proposed method, we conducted experimental tests on standard image datasets such as Extended Yale B, COIL20, COIL100, and ORL. The experimental results show that the performance of the proposed method is better than the existing traditional subspace clustering algorithm and deep clustering algorithm.



# Platform

This code was developed and tested with:

```python
networkx          2.6.3
Pillow            8.4.0
torch             1.7.0+cu101
torchvision       0.8.1+cu101

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn==0.22
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple munkres==1.1.4
```



# citation

if you use this code for your research, please cite our paper:

@article{DSCNSS,
title={ Deep subspace image clustering network with self-expression and self-supervision },
author={Chao Chen, Hu Lu, Hui Wei, Xia Geng },
journal={Applied Intelligence},
year={2022},
}







