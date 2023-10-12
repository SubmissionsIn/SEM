## SEM: SElf-weighted Multi-view contrastive learning with reconstruction regularization

Prepare a multi-view/modal dataset, for example:

The format of a multi-view dataset ($N$ samples and $V$ views) should be $\{\mathbf{X}^1, \mathbf{X}^2, \dots, \mathbf{X}^v, \dots, \mathbf{X}^V, \mathbf{Y}\}$, where the $v$-th view data is $\mathbf{X}^v\in \mathbb{R}^{N\times d_v}$ and the class label is $\mathbf{Y}\in \mathbb{R}^{N\times 1}$ (The label is leveraged to evaluate the performance of representation learning in unsupervised settings). This type of data is suitable for fully connected neural networks, otherwise the model in "network.py" needs to be modified.

The public datasets and our trained models are available at **[Download](https://drive.google.com/drive/folders/1JBhb66b_z2wB4xWcuvrRvaINhgeCxiDS?usp=drive_link)** or **[国内下载源](https://pan.baidu.com/s/1m8Vi3RShRMDUTjs-TZCiAQ?pwd=0928)**.


Requirements:

    python==3.7.11
    pytorch==1.9.0
    numpy==1.20.1
    scikit-learn==0.22.2.post1
    scipy==1.6.2

To test the trained model, run:
```bash
python test.py
```
    

To train a new model, run:
```bash
python train.py
```
