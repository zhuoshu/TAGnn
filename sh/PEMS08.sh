python generate_datasets.py --dataset_name PEMS08 --input_length 12 --predict_length 12 --save 1
python train.py --dataset_name PEMS08 --input_length 12 --predict_length 12 --hid_dim 64 --tcn_kernel_size 2 --TimeEncodingType 3 --addLatestX 1 --hasCross 1