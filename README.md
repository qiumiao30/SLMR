# SLMR
# Introduction
pass

# Geting Started
To clone this repo:
```bash
git clone https://github.com/qiumiao30/UMDCR.git && cd UMDCR
```

## 1. get data
```bash

```

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

pass
