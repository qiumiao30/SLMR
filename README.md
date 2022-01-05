# SLMR
# Introduction
**Upload the code after the paper is accepted**

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
python data_preprocess.py --dataset $dataset_name$
```
`$dataset$` is one of SWAT, MSL, SMAP et al.

for example:

```python
python data_preprocess.py --dataset swat
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
python main.py
```

## 6. visualization 
pass

# Model Overview

pass

# Acknowledge

**DAGMM:** [code](https://github.com/tnakae/DAGMM)
**LSTM-VAE:** [code](https://github.com/Danyleb/Variational-Lstm-Autoencoder)
**MAD-GAN:** [code](https://github.com/LiDan456/MAD-GANs)
**MTAD-GAT:** [code](https://github.com/mangushev/mtad-gat)
**USAD:** [code](https://github.com/finloop/usad-torchlightning)
**OmniAnomaly:** [code](https://github.com/NetManAIOps/OmniAnomaly)
