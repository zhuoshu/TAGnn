# TAGnn
This is a Pytorch implementation of TAGnn: Time Adjoint Graph Neural Network for Traffic Forecasting. (DASFAA 2023)

## Data
The datasets are provided at [ASTGNN](https://github.com/guoshnBJTU/ASTGNN) or [STSGCN](https://github.com/Davidham3/STSGCN), and the `PEMS0X.npz` files should be put into the corresponding `data/` folder.

## Requirements
Python 3.9.12, torch 1.11.0, numpy 1.21.5

## Usage
The commands on PEMS08 are presented as example.

Step 1: Generate Datasets (using one-hour history data to predict data in the next hour)
```python
python generate_datasets.py --dataset_name PEMS08 --input_length 12 --predict_length 12 --save 1
```

Step 2: Train and Test
```python
python train.py --dataset_name PEMS08 --input_length 12 --predict_length 12
```

The hyper-parameter settings for all datasets are given in `sh/` folder.