# SLMR

[**paper in arixv**](https://arxiv.org/abs/2208.09240)

# Introduction

Anomaly detection of multivariate time series is meaningful for system behavior monitoring. This paper proposes an anomaly detection method based on unsupervised Short- and Long-term Mask Representation learning (SLMR). The main idea is to extract short-term local dependency patterns and long-term global trend patterns of the multivariate time series by using multi-scale residual dilated convolution and Gated Recurrent Unit(GRU) respectively. Furthermore, our approach can comprehend temporal contexts and feature correlations by combining spatial-temporal masked self-supervised representation learning and sequence split. It considers the importance of features is different, and we introduce the attention mechanism to adjust the contribution of each feature. Finally, a forecasting-based model and a reconstruction-based model are integrated to focus on single timestamp prediction and latent representation of time series. Experiments show that the performance of our method outperforms other state-of-the-art models on three real-world datasets. Further analysis shows that our method is good at interpretability.

# Model Overview

![model](https://github.com/qiumiao30/SLMR/blob/master/image/model.png)

# Geting Started
To clone this repo:
```bash
git clone https://github.com/qiumiao30/SLMR.git && cd SLMR
```

## 1. get data

- **SWaT:** [SWaT Dataset Download](https://itrust.sutd.edu.sg/itrust-labs_datasets/), [Dataset Introduce](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat/)
- **MSL & SMAP:** [Dataset Download and Introduction](https://github.com/khundman/telemanom)

## 2. Install Dependencies(Recomend Virtualenv)

- python>=3.7
- torch>=1.9

```python
pip install -r requirements.txt
```

## 3. dataset preprocess

```python
python preprocess.py --dataset $dataset_name$
```
`$dataset$` is one of SWAT, MSL, SMAP et al.

for example:

```python
python preprocess.py --dataset swat
```

## 4. Params

> - --dataset :  default "swat".
> - --lookback : Windows size, default 10.
> - --normalize : Whether to normalize, default True.
> - --epochs : default 10
> - --bs : Batch Size, default 256
> - --init_lr : init learning rate, default 1e-3
> - --val_split : val dataset, default 0.1
> - --dropout : 0.3

## 5. run

```python
python train.py --Params "value" --Parmas "value" ......
```

## 6. visualization 
![vis](https://github.com/qiumiao30/SLMR/blob/master/image/vis.png)

# Acknowledge
1. Other Advanced Methods
- **DAGMM:** [code](https://github.com/tnakae/DAGMM)
- **LSTM-VAE:** [code](https://github.com/Danyleb/Variational-Lstm-Autoencoder)
- **MAD-GAN:** [code](https://github.com/LiDan456/MAD-GANs)
- **MTAD-GAT:** [code](https://github.com/mangushev/mtad-gat)
- **USAD:** [code](https://github.com/finloop/usad-torchlightning)
- **OmniAnomaly:** [code](https://github.com/NetManAIOps/OmniAnomaly)

# cite

```
@InProceedings{10.1007/978-981-99-1645-0_42,
author="Miao, Qiucheng
and Xu, Chuanfu
and Zhan, Jun
and Zhu, Dong
and Wu, Chengkun",
title="An Unsupervised Short- and Long-Term Mask Representation for Multivariate Time Series Anomaly Detection",
booktitle="Neural Information Processing",
year="2023",
publisher="Springer Nature Singapore",
address="Singapore",
pages="504--516",
isbn="978-981-99-1645-0"
}


