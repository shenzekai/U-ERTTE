import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import argparse
import time
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def picp(y_true, upper_bound, lower_bound):
    """
    y_true : B, N, T, D
    upper_bound, lower_bound : B, N, T, D
    """
    # 转换 torch.where 为 numpy 的 np.where
    result1 = np.where(y_true < upper_bound, np.ones_like(y_true), np.zeros_like(y_true))
    result2 = np.where(y_true > lower_bound, np.ones_like(y_true), np.zeros_like(y_true))

    # 计算 recalibrate_rate
    recalibrate_rate = np.sum(result1 * result2) / np.prod(y_true.shape)

    return recalibrate_rate * 100


def mpiw(y_true, upper_bound, lower_bound):
    """
    y_true,y_pred : B, N, T, D
    sigma : B, N, T, D
    """
    MPIW = upper_bound - lower_bound
    MPIW = np.sum(MPIW) / np.prod(y_true.shape)

    return MPIW





def mape_(y_hat, y, length):
    length = length.cpu().numpy().tolist()
    weight = [1 if i > 40 else 1.2 for i in length]
    weight = torch.tensor(weight, device=device)
    l = torch.abs(y_hat - y) / y
    l *= weight
    return l.mean()


def SR(y_true, y_pred):
    MAPE = abs(y_pred - y_true) / y_true
    sr = np.sum(MAPE <= 0.1) / np.prod(y_true.shape)*100
    return sr


def mis(y_true, upper_bound, lower_bound):
    """
    y_pred: T B V F
    y_true: T B V
    """

    # pho = 0.05 # 置信度 97.5 pho/2
    pho = 0.10
    #     loss0 = torch.abs(y_pred.T[2].T - y_true) # MAE
    loss1 = np.maximum(upper_bound - lower_bound, np.array([0.]))  # u-l
    loss2 = np.maximum(lower_bound - y_true, np.array([0.])) * 2 / pho  # l-y 哪些下界值超了
    loss3 = np.maximum(y_true - upper_bound, np.array([0.])) * 2 / pho  # y-u 哪些上界值小了
    #     print(loss1,loss2,loss3)
    loss = loss1 + loss2 + loss3
    return loss.mean()


def QICE(y_true, y_upper, y_low, y_pred):
    Q1 = ((y_low - y_true) > 0).astype(int)
    q21 = ((y_true - y_low) > 0).astype(int)
    q22 = ((y_pred - y_true) > 0).astype(int)
    Q2 = q21 * q22
    q31 = ((y_true - y_pred) > 0).astype(int)
    q32 = ((y_upper - y_true) > 0).astype(int)
    Q3 = q31 * q32
    Q4 = ((y_true - y_upper) > 0).astype(int)
    PQ1 = np.absolute(np.sum(Q1) / len(y_true) - 0.05)
    PQ2 = np.absolute(np.sum(Q2) / len(y_true) - 0.45)
    PQ3 = np.absolute(np.sum(Q3) / len(y_true) - 0.45)
    PQ4 = np.absolute(np.sum(Q4) / len(y_true) - 0.05)
    # print(PQ1, PQ2, PQ3, PQ4)
    qice = PQ1 + PQ2 + PQ3 + PQ4
    return qice


def UncertaintyPercentage(sigma, real):
    UP = sigma / real
    return UP.mean()


def quantile_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    # quantiles = [0.025, 0.5, 0.975]  # 0.5 均值
    # quantiles = [0.10, 0.5, 0.90]  # 0.5 均值
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, i]
        errors = errors * mask
        errors[errors != errors] = 0
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return loss


def maemis_loss(y_pred, y_true):
    """
    y_pred: T B V F
    y_true: T B V
    """
    mask = (y_true != 0).float()
    mask /= mask.mean()
    # pho = 0.05 # 置信度 97.5 pho/2
    pho = 0.10
    loss0 = torch.abs(y_pred.T[2].T - y_true)
    loss1 = torch.max(y_pred[:, 0] - y_pred.T[1].T, torch.tensor([0.]).to(y_true.device))

    loss2 = torch.max(y_pred.T[1].T - y_true, torch.tensor([0.]).to(y_true.device)) * 2 / pho
    loss3 = torch.max(y_true - y_pred[:, 0], torch.tensor([0.]).to(y_true.device)) * 2 / pho
    loss = loss0 + loss1 + loss2 + loss3
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def calculate_metrics(label, predicts, loss):
    metrics = {}
    if loss == 'maemis':
        upper_bound = predicts[:, 0]
        lower_bound = predicts[:, 1]
        predicts = predicts[:, 2]
    elif loss == 'quantile':
        upper_bound = predicts[:, 2]
        lower_bound = predicts[:, 0]
        predicts = predicts[:, 1]
    # gap = calculate_gap(label, predicts)
    # print("comparison", np.sum(gap > 0), np.sum(gap <= 0))
    # point mertics
    metrics['mape'] = mape(label, predicts)
    metrics['mse'] = mse(label, predicts)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mae(label, predicts)
    metrics['sr'] = SR(label, predicts)
    # interval metrics
    if loss not in ["mape", "MAE"]:
        metrics['picp'] = picp(label, upper_bound, lower_bound)
        metrics['mis'] = mis(label, upper_bound, lower_bound)
        metrics['mpiw'] = mpiw(label, upper_bound, lower_bound)
    return metrics

# 计算吞吐率
def calculate_throughput(total_samples, process_time):
    return total_samples / process_time

def calculate_all_metrics(loss, label, predicts, mid_label, mid_predicts, er_targets, er_predicts, is_in=False):
    metrics = {}
    def add_prefix(metrics_dict, prefix):
        return {f"{prefix}_{k}": v for k, v in metrics_dict.items()}
    # Calculate and add metrics for Full data
    metrics.update(add_prefix(calculate_metrics(label, predicts, loss), 'val'))
    metrics.update(add_prefix(calculate_metrics(mid_label, mid_predicts, loss), 'mid'))
    # Calculate and add metrics for mid data and remain data
    if loss == 'quantile' or loss == 'maemis':
        if is_in and len(er_predicts) == 0:
            for metric in ['mape', 'mse', 'mae', 'sr', 'picp', 'mis', 'mpiw']:
                metrics[f're_{metric}'] = 0
                metrics['re_rmse'] = 0
        else:
            metrics.update(add_prefix(calculate_metrics(er_targets, er_predicts, loss), 're'))
    print_results(metrics, loss, er_predicts, is_in)
    return metrics

def print_results(metrics, loss, er_predicts, is_in=False):
    # Function to print results
    if loss not in ["mape", "MAE"]:
        print('MAPE:%.3f RMSE: %.3f \tMSE:%.2f\tMAE:%.2f\tSR:%.3f\tPICP:%.3f\tMIS:%.3f\tMPIW：%.3f\t' % (
            metrics['val_mape'] * 100, metrics['val_rmse'], metrics['val_mse'], metrics['val_mae'], metrics['val_sr'], metrics['val_picp'],
            metrics['val_mis'], metrics['val_mpiw']))
        print("PR-TTE MID RESULTS:")
        print('MAPE:%.3f\tMAE:%.2f\tRMSE:%.2f\tSR:%.2f' % (metrics['mid_mape'] * 100, metrics['mid_mae'], metrics['mid_rmse'], metrics['mid_sr']))
        print('PICP:%.3f\tMIS:%.3f\tMPIW：%.3f' % (metrics['mid_picp'], metrics['mid_mis'], metrics['mid_mpiw']))
        if len(er_predicts) > 0:
            print("ER-TTE RESULTS:")
            if is_in:
                print("Don't Need %d" % len(er_predicts))
            print('MAPE:%.3f\tMAE:%.3f\tRMSE:%.2f\tSR:%.3f' % (metrics['re_mape'] * 100, metrics['re_mae'], metrics['re_rmse'], metrics['re_sr']))
            print('PICP:%.3f\tMIS:%.3f\tMPIW：%.3f' % (metrics['re_picp'], metrics['re_mis'], metrics['re_mpiw']))
    else:
        print('Full MAPE:%.3f\tSR:%.3f\t rmse: %.3f\tMSE:%.2f\tMAE:%.2f' % (
        metrics['val_mape'] * 100, metrics['val_sr'], metrics['val_rmse'], metrics['val_mse'], metrics['val_mae']))
        print("PR-TTE MID RESULTS:")
        print('MAPE:%.3f\tMAE:%.2f\tSR:%.2f' % (metrics['mid_mape'] * 100, metrics['mid_mae'], metrics['mid_sr']))
        print("ER-TTE RESULTS:")
        print('MAPE:%.3f\tMAE:%.3f\tSR:%.3f' % (metrics['re_mape'] * 100, metrics['re_mae'], metrics['re_sr']))

