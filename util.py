import numpy as np
import os
import torch

class MinMaxScaler():
    """
    normalize the input to [-1, 1]
    """

    def __init__(self, max, min):
        self.max = max
        self.min = min

        print('_max.shape:', max.shape)
        print('_min.shape:', min.shape)


    def transform(self, data):
        x=1.*(data - self.min) / (self.max - self.min)
        return 2. *x-1.

    def inverse_transform(self, data):
        x= (data+1.)/2.
        x=x*(self.max - self.min) + self.min
        return x

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataLoader_time(object):
    def __init__(self, x, y, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        xs, xtod, xdow = x
        ys = y

        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            xtod_padding = np.repeat(xtod[-1:], num_padding, axis=0)
            xdow_padding = np.repeat(xdow[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            xtod = np.concatenate([xtod, xtod_padding], axis=0)
            xdow = np.concatenate([xdow, xdow_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.xtod = xtod
        self.xdow = xdow
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        xtod = self.xtod[permutation]
        xdow = self.xdow[permutation]
        self.xs = xs
        self.xtod = xtod
        self.xdow = xdow
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                xtod_i = self.xtod[start_ind: end_ind, ...]
                xdow_i = self.xdow[start_ind: end_ind, ...]

                y_i = self.ys[start_ind: end_ind, ...]
                yield ([x_i, xtod_i, xdow_i], y_i)
                self.current_ind += 1

        return _wrapper()

def load_dataset_time(dataset_name, input_length, predict_length, batch_size, valid_batch_size=None, test_batch_size=None,scalertype='minmax',fillZero=False):
    data = {}
    dataset_dir = f'./data/{dataset_name}'
    tip=f'{input_length}to{predict_length}'
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, dataset_name + '_' + category + f'_{tip}.npz'))
        data['x_' + category] = cat_data['x'][...,0:1] #flow
        data['y_' + category] = cat_data['y'][...,0:1]
        data['x_tod_' + category] = cat_data['x_tod']
        data['x_dow_' + category] = cat_data['x_dow']

    print("data['x_train'][..., 0].shape=",data['x_train'][..., 0].shape)
    
    assert scalertype in ['minmax','zscore']
    if scalertype=='minmax':
        _max = data['x_train'][..., 0:1].max()
        _min = data['x_train'][..., 0:1].min()
        scaler=MinMaxScaler(max=_max,min=_min)
    else:
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader_time([data['x_train'], data['x_tod_train'], data['x_dow_train']],
                                      data['y_train'][..., 0:1], batch_size)
    data['val_loader'] = DataLoader_time([data['x_val'], data['x_tod_val'], data['x_dow_val']],
                                     data['y_val'][..., 0:1], valid_batch_size)
    data['test_loader'] = DataLoader_time([data['x_test'], data['x_tod_test'], data['x_dow_test']], 
                                    data['y_test'][..., 0:1], test_batch_size)

    data['scaler'] = scaler
    print('train_x.shape:',data['x_train'].shape)
    print('val_x.shape:',data['x_val'].shape)
    print('test_x.shape:',data['x_test'].shape)

    num_samples = data['x_train'].shape[0]
    seq_len = data['x_train'].shape[1]
    num_nodes = data['x_train'].shape[2]
    in_dim = data['x_train'].shape[-1]
    print('num_nodes:', num_nodes)

    return data, num_nodes, in_dim

def mae(preds,labels):
    loss = torch.mean(torch.abs(preds - labels))
    return loss
def mse(preds,labels):
    loss = torch.mean(torch.square(preds - labels))
    return loss

def rmse(preds,labels):
    loss = torch.mean(torch.square(preds - labels))
    return loss**0.5

def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def metric(predict,real,masked_all=False):
    if masked_all:
        nmae = masked_mae(predict, real, 0.0).item()
        nrmse = masked_rmse(predict, real, 0.0).item()
    else:
        nmae = mae(predict, real).item()
        nrmse = rmse(predict, real).item()
    mape = masked_mape(predict, real, 0.0).item()
    return nmae,mape,nrmse

