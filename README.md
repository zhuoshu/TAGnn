# TAGnn
This is a Pytorch implementation of TAGnn: Time Adjoint Graph Neural Network for Traffic Forecasting. (DASFAA 2023) [[Paper](https://dl.acm.org/doi/10.1007/978-3-031-30637-2_24)]

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

## Cite
If you find this project helpful, please cite us:
```bibtex
@inproceedings{10.1007/978-3-031-30637-2_24,
author = {Zheng, Qi and Zhang, Yaying},
title = {TAGnn: Time Adjoint Graph Neural Network for&nbsp;Traffic Forecasting},
year = {2023},
url = {https://doi.org/10.1007/978-3-031-30637-2_24},
doi = {10.1007/978-3-031-30637-2_24},
booktitle = {Database Systems for Advanced Applications: 28th International Conference, DASFAA 2023, Tianjin, China, April 17–20, 2023, Proceedings, Part I},
pages = {369–379}
}
```
