import argparse
import numpy as np
import os

def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, dataset_name='PEMS08',slice_size_per_day=288
):
    num_samples, num_nodes, in_dim = data.shape
    print(data.shape)

    slice_start_dict={'PEMS03':0,'PEMS04':0,'PEMS07':0,'PEMS08':0,
                    'HZME_INFLOW':0,'HZME_OUTFLOW':0}
    slice_start=slice_start_dict[dataset_name]  # 00:00
    tod = [(i+slice_start) % slice_size_per_day for i in range(num_samples)]
    t_of_d = np.array(tod)
    t_of_d=np.expand_dims(t_of_d,1)
    print('t_of_d.shape:',t_of_d.shape)
    # print(t_of_d)

    start_date_dict={ 
        'PEMS03':'2018-09-01','PEMS04':'2018-01-01',
        'PEMS07':'2017-05-01','PEMS08':'2016-07-01',
        'HEME':'2019-01-01 00:00 to 2019-01-27 00:00',
    }
    # the day of the week of the beginning day
    day_start_dict={'PEMS03':5,'PEMS04':0,'PEMS07':0,'PEMS08':4,
                    'HZME_INFLOW':1,'HZME_OUTFLOW':1}
    day_start=day_start_dict[dataset_name] 
    
    dow = [(i // slice_size_per_day + day_start) % 7 for i in range(num_samples)]
    d_of_w = np.array(dow)
    d_of_w=np.expand_dims(d_of_w,1)


    print('d_of_w.shape:',d_of_w.shape)
    # print(d_of_w)
    
    x, y = [], []
    x_tod = []
    x_dow = []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        x_tod.append(t_of_d[t + x_offsets, ...])
        x_dow.append(d_of_w[t + x_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    x_tod = np.stack(x_tod, axis=0)
    x_dow = np.stack(x_dow, axis=0)
    return x, y, x_tod, x_dow


def generate_train_val_test(dataset_name, graph_signal_matrix_filename, input_length, predict_length, output_dir,save=False):
    y_start = 1
    ori=np.load(graph_signal_matrix_filename)
    data = ori['data']
    print(data.shape)
    x_offsets = np.sort(np.concatenate((np.arange(-(input_length - 1), 1, 1),)))
    print('x_offsets:',x_offsets)
    # Predict the next one hour
    y_offsets = np.sort(np.arange(y_start, (predict_length + 1), 1))
    print('y_offsets:',y_offsets)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)

    x, y, x_tod, x_dow= generate_graph_seq2seq_io_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        dataset_name=dataset_name
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape) 
    print('x_tod.shape: ',x_tod.shape)
    print('x_dow.shape: ',x_dow.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    
    split_line1 = int(num_samples * 0.6)
    split_line2 = int(num_samples * 0.8)

    x_train, y_train = x[:split_line1], y[:split_line1]
    x_val, y_val = (
        x[split_line1: split_line2],
        y[split_line1: split_line2],
    )
    x_test, y_test = x[split_line2:], y[split_line2:]

   
    x_tod_train = x_tod[:split_line1]
    x_tod_val = x_tod[split_line1: split_line2]
    x_tod_test = x_tod[split_line2:]

    x_dow_train = x_dow[:split_line1]
    x_dow_val = x_dow[split_line1: split_line2]
    x_dow_test = x_dow[split_line2:]


    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _x_tod =locals()['x_tod_'+cat]
        _x_dow =locals()['x_dow_'+cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape) 
        print(cat,"x_tod:",_x_tod.shape)
        print(cat,"x_dow:",_x_dow.shape)
        if save:
            np.savez_compressed(
                os.path.join(output_dir, f"{dataset_name}_{cat}_{input_length}to{predict_length}.npz"),
                x=_x,
                y=_y,
                x_tod=_x_tod,
                x_dow=_x_dow,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )
    if save:
        with open(os.path.join(output_dir, f"{dataset_name}_{input_length}to{predict_length}_description.txt"), 'a+') as f:
            f.write(f"x.shape={x.shape}, y.shape={y.shape}\n")
            f.write(f"x_tod.shape={x_tod.shape}\n")
            f.write(f"x_dow.shape={x_dow.shape}\n")
            f.write(f'train:\n')
            f.write(f"\tx_train.shape={x_train.shape}, y_train.shape={y_train.shape}\n")
            f.write(f"\tx_tod_train.shape={x_tod_train.shape}\n")
            f.write(f"\tx_dow_train.shape={x_dow_train.shape}\n")
            f.write(f'val:\n')
            f.write(f"\tx_val.shape={x_val.shape}, y_val.shape={y_val.shape}\n")
            f.write(f"\tx_tod_val.shape={x_tod_val.shape}\n")
            f.write(f"\tx_dow_val.shape={x_dow_val.shape}\n")
            f.write(f'test:\n')
            f.write(f"\tx_test.shape={x_test.shape}, y_test.shape={y_test.shape}\n")
            f.write(f"\tx_tod_test.shape={x_tod_test.shape}\n")
            f.write(f"\tx_dow_test.shape={x_dow_test.shape}\n")




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="PEMS08", help="Dataset name.")
    parser.add_argument("--input_length", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--predict_length", type=int, default=12, help="Sequence Length.",)    
    parser.add_argument("--save", type=int, default=0, help="save", )

    args = parser.parse_args()

    graph_signal_matrix_filename=f"./data/{args.dataset_name}.npz"
    output_dir=f"./data/{args.dataset_name}"

    if os.path.exists(output_dir):
        reply = str(input(f'{output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(output_dir)
    generate_train_val_test(args.dataset_name,graph_signal_matrix_filename,args.input_length,args.predict_length,output_dir,save=bool(args.save))