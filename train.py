import torch
import numpy as np
import argparse
import time
import util
import csv
from torch import nn
import torch.optim as optim
from model.TAGnn import TAGnn
import os
import torch.nn.functional as F
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--dataset_name', type=str, default='PEMS04', help='')
parser.add_argument('--input_length', type=int, default=12, help='')
parser.add_argument('--predict_length', type=int, default=12, help='')
parser.add_argument('--hid_dim', type=int, default=64, help='')
parser.add_argument('--tcn_kernel_size', type=int, default=3, help='')

parser.add_argument('--TimeEncodingType', type=int, default=3,
                    help='3: use Day and Slice information, 2: use Day only, 1:use Slice only, 0:not use time prior')
parser.add_argument('--addLatestX', type=int, default=1,
                    help='add the latest input time slice into the prediction')
parser.add_argument('--hasCross', type=int, default=1,
                    help='connect each time slice with the latest one')

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--early_stop_step', type=int, default=20)

parser.add_argument('--seed', type=int, default=7, help='random seed')
parser.add_argument('--save', type=str, default='', help='save path')
parser.add_argument('--tip', type=str, default='TAGnn_final', help='tip')

expid = time.strftime("%m%d%H%M", time.localtime())
args = parser.parse_args()


class trainer():
    def __init__(self, scaler,
                 in_dim, input_length, predict_length, num_nodes,
                 lrate, wdecay, device,
                 hid_dim=64, tcn_kernel_size=3,
                 TimeEncodingType=3, addLatestX=True, hasCross=True,
                 ExcludeZero=False):
        self.model = TAGnn(input_length=input_length, predict_length=predict_length, num_of_vertices=num_nodes, num_of_features=in_dim,
                           hid_dim=hid_dim, tcn_kernel_size=tcn_kernel_size,
                           TimeEncodingType=TimeEncodingType, addLatestX=addLatestX, hasCross=hasCross)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model.to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = nn.L1Loss()
        self.scaler = scaler
        self.clip = 5
        self.excludeZero = ExcludeZero

    def train(self, input, real_val):  # input(B,C,N,T),real_val(B,N,T)
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)  # (B,T,N,1)
        output = output.transpose(1, 3)  # (B,1,N,T)
        real = torch.unsqueeze(real_val, dim=1)  # (B,1,N,T)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        if self.excludeZero:
            mae = util.masked_mae(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
        else:
            mae = util.mae(predict, real).item()
            rmse = util.rmse(predict, real).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)  # (B,C,N,T)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real)
        if self.excludeZero:
            mae = util.masked_mae(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
        else:
            mae = util.mae(predict, real).item()
            rmse = util.rmse(predict, real).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse


def checkLoss(loss):
    if loss > 10000 or loss is None or loss != loss:
        print('iter:', iter)
        print('loss:', loss)
        print('>10000:', loss > 10000)
        print('is None:', loss is None)
        print('is Nan:', loss != loss)
        return True
    return False


def main():
    # load data
    model_name = f'TAGnn_final_{args.dataset_name}'
    device = torch.device(args.device)

    dataloader, num_nodes, in_dim = util.load_dataset_time(args.dataset_name, args.input_length, args.predict_length,
                                                           args.batch_size, args.batch_size, args.batch_size)
    adj_mx = [np.diag(np.ones(num_nodes)).astype(np.float32)]
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    scaler = dataloader['scaler']
    expid = time.strftime("%m%d%H%M", time.localtime())
    save_path = f'./{model_name}checkpoint/' + \
        args.dataset_name + f"/{model_name}_exp" + str(expid) + "/"
    print(args)

    in_dim = 1
    excludeZero = False
    engine = trainer(scaler, in_dim, args.input_length, args.predict_length, num_nodes,
                     lrate=args.learning_rate, wdecay=args.weight_decay, device=device,
                     hid_dim=args.hid_dim, tcn_kernel_size=args.tcn_kernel_size,
                     TimeEncodingType=args.TimeEncodingType,addLatestX=bool(args.addLatestX),hasCross=bool(args.hasCross),
                     ExcludeZero=excludeZero)

    print("start training...", flush=True)
    all_start_time = time.time()
    his_loss = []
    trainloss_record = []
    val_time = []
    train_time = []
    best_validate_loss = np.inf
    validate_score_non_decrease_count = 0

    log_in_train_details = []

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x, xtod, xdow = x
            x = x[:, :, :, 0:1]
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            xtod = torch.LongTensor(xtod).to(device)
            trainxtod = F.one_hot(xtod, num_classes=288).squeeze(
                2).float()  # (B,12,288)
            xdow = torch.LongTensor(xdow).to(device)
            trainxdow = F.one_hot(xdow, num_classes=7).squeeze(
                2).float()  # (B,12,7)

            trainx = [trainx, trainxtod, trainxdow]
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f},Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(
                    iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]), flush=True)
                if checkLoss(metrics[0]):
                    sys.exit('Please try again.')
        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            x, xtod, xdow = x
            x = x[:, :, :, 0:1]
            valx = torch.Tensor(x).to(device)
            valx = valx.transpose(1, 3)
            valy = torch.Tensor(y).to(device)
            valy = valy.transpose(1, 3)

            xtod = torch.LongTensor(xtod).to(device)
            valxtod = F.one_hot(xtod, num_classes=288).squeeze(
                2).float()  # (B,12,288)
            xdow = torch.LongTensor(xdow).to(device)
            valxdow = F.one_hot(xdow, num_classes=7).squeeze(
                2).float()  # (B,12,7)

            valx = [valx, valxtod, valxdow]

            metrics = engine.eval(valx, valy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        trainloss_record.append(
            [mtrain_loss, mtrain_mae, mtrain_mape, mtrain_mape, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse])
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
        log += 'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                         mvalid_rmse, (t2 - t1)), flush=True)
        log_in_train_details.append([i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                                     mvalid_rmse, (t2 - t1)])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if best_validate_loss > mvalid_loss:
            best_validate_loss = mvalid_loss
            validate_score_non_decrease_count = 0
            torch.save(engine.model.state_dict(), save_path + "best.pth")
            print('got best validation result:',
                  mvalid_loss, mvalid_mape, mvalid_rmse)
        else:
            validate_score_non_decrease_count += 1

        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break

    print(
        "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    training_time = (time.time() - all_start_time) / 60

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(save_path + "best.pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    test_start_time = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x, xtod, xdow = x
        x = x[:, :, :, 0:1]
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)

        xtod = torch.LongTensor(xtod).to(device)
        testxtod = F.one_hot(xtod, num_classes=288).squeeze(2).float()  # (B,12,288)
        xdow = torch.LongTensor(xdow).to(device)
        testxdow = F.one_hot(xdow, num_classes=7).squeeze(2).float()  # (B,12,7)

        testx = [testx, testxtod, testxdow]
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)

    yhat = yhat[:realy.size(0), ...]

    inference_time = time.time() - test_start_time
    print('yhat.shape=', yhat.shape)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real, masked_all=excludeZero)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    MAE = np.mean(amae)
    MAPE = np.mean(amape)
    RMSE = np.mean(armse)
    print(log.format(MAE, MAPE, RMSE))
    
    result_txt_path = f'{model_name}_Results.txt'
    result_csv_path = f'{model_name}_Results.csv'
    log_csv_path = f'Log_Train_{model_name}.csv'
    train_metric = trainloss_record[bestid][0:4]
    valid_metric = trainloss_record[bestid][4:]
    with open(result_csv_path, 'a+', newline='')as f0:
        f_csv = csv.writer(f0)
        row = [expid, args.dataset_name,args.tip, MAE, MAPE*100, RMSE, train_metric, valid_metric, 
        [amae[2],amape[2]*100,armse[2]],
        [amae[5],amape[5]*100,armse[5]],
        [amae[11],amape[11]*100,armse[11]]        
        ]
        f_csv.writerow(row)

    with open(log_csv_path, 'a+', newline='')as flog:
        flog_csv = csv.writer(flog)
        flog_csv.writerow([expid, args.dataset_name, args.tip])
        for it in log_in_train_details:
            flog_csv.writerow(it)

    with open(result_txt_path, 'a+') as f:
        f.write(
            f"【{expid}】{args.tip} {args.dataset_name} epoch={len(trainloss_record)} bestid={bestid}:")
        f.write(f'\n{args} in-dim={in_dim}')
        f.write(
            f'\ntraining_time={training_time}min, inference_time={inference_time}s')
        f.write('\nMAE_list:')
        for id, it in enumerate(amae):
            f.write('%.4f\t' % it)
        f.write('\nMAPE_list:')
        for id, it in enumerate(amape):
            f.write('%.4f\t' % it)
        f.write('\nRMSE_list:')
        for id, it in enumerate(armse):
            f.write('%.4f\t' % it)
        f.write('\ntrain metric:')
        for id, it in enumerate(train_metric):
            f.write('%.4f\t' % it)
        f.write('\nvalid metric:')
        for id, it in enumerate(valid_metric):
            f.write('%.4f\t' % it)
        f.write('\nOn average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}\n\n'.format(MAE,
                                                                                                                   MAPE,
                                                                                                                   RMSE))
        f.flush()


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
