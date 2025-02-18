import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import time
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from prefetch_generator import BackgroundGenerator
from torch.cuda.amp import autocast as autocast, GradScaler


# 构建Dataset
class TTEDataset(torch.utils.data.Dataset):
    def __init__(self, seq_data, FLAGS):
        mode = FLAGS.er_mode
        all_num = []
        all_mid_num = []
        all_re_num = []
        all_mid_start = []
        all_id = []
        all_id_re = []
        all_time = []
        all_time_re = []
        all_flow = []
        all_flow_re = []
        all_linkdistance = []
        all_linkdistance_re = []
        all_label = []
        all_mid_label = []
        all_re_label = []
        all_cross = []
        all_oneway = []
        all_oneway_re = []
        all_reversed = []
        all_reversed_re = []
        all_highway = []
        all_highway_re = []
        all_lane = []
        all_lane_re = []
        drivers_num = FLAGS.drivers_num
        print(seq_data.shape)

        for i in range(len(seq_data)):
            ids = seq_data[i][4]
            all_id.append(ids)  # link id

            # 提取link id
            id_re = seq_data[i][4][seq_data[i][25 + mode*2]:]   # traveled link id
            all_id_re.append(id_re)  # link id

            all_mid_start.append(id_re[0])  # start index of mid link
            time = seq_data[i][11]
            time_re = seq_data[i][11][seq_data[i][25 + mode*2]:]
            all_time.append(time)  # list
            all_time_re.append(time_re)  # list

            flow = seq_data[i][12]
            flow_re = seq_data[i][12][seq_data[i][25+mode*2]:]
            all_flow.append(flow)  # list
            all_flow_re.append(flow_re)  # list

            linkdistance = seq_data[i][2]
            linkdistance_re = seq_data[i][2][seq_data[i][25+mode*2]:]
            all_linkdistance.append(linkdistance)  # list
            all_linkdistance_re.append(linkdistance_re)  # list

            highway = seq_data[i][15]
            highway_re = seq_data[i][15][seq_data[i][25+mode*2]:]
            all_highway.append(highway)  # list
            all_highway_re.append(highway_re)  # list

            lane = seq_data[i][16]
            lane_re = seq_data[i][16][seq_data[i][25+mode*2]:]
            all_lane.append(lane)  # list
            all_lane_re.append(lane_re)  # list

            oneway = seq_data[i][13]
            oneway_re = seq_data[i][13][seq_data[i][25+mode*2]:]
            all_oneway.append(oneway)  # list
            all_oneway_re.append(oneway_re)  # list

            reversed = seq_data[i][14]
            reversed_re = seq_data[i][14][seq_data[i][25+mode*2]:]
            all_reversed.append(reversed)  # list
            all_reversed_re.append(reversed_re)  # list

            all_num.append(seq_data[i][5])  # link num porto-1记住
            all_cross.append(seq_data[i][6])  # cross num
            all_mid_num.append(seq_data[i][25+mode*2])  # mid link num
            all_re_num.append(seq_data[i][26+mode*2])  # re link num
            all_label.append(seq_data[i][8])  # label
            all_mid_label.append(seq_data[i][17+mode*2])  # mid_label
            all_re_label.append(seq_data[i][18+mode*2])  # re_label
        self.all_num = all_num
        self.all_mid_num = all_mid_num
        self.all_re_num = all_re_num
        self.all_mid_start = all_mid_start

        # link 平均通行时间
        self.all_real = all_time
        self.all_real_re = all_time_re

        # 流量特征
        self.all_flow = all_flow
        self.all_flow_re = all_flow_re

        # 行程中的路段长度
        self.all_linkdistance = all_linkdistance
        self.all_linkdistance_re = all_linkdistance_re

        # 行程中的路段ID序列 70% 40% 10%
        self.all_id = all_id
        self.all_id_re = all_id_re

        self.all_highway = all_highway
        self.all_highway_re = all_highway_re

        self.all_lane = all_lane
        self.all_lane_re = all_lane_re

        self.all_oneway = all_oneway
        self.all_oneway_re = all_oneway_re

        self.all_reversed = all_reversed
        self.all_reversed_re = all_reversed_re

        self.targets = all_label
        self.mid_targets = all_mid_label
        self.re_targets = all_re_label

        self.departure = seq_data[:, 10]  # slice_window
        self.driver_id = seq_data[:, 1] # driver_id
        self.weekday = seq_data[:, 9]  # weekday
        self.distance = seq_data[:, 3]  # distance
        # wide_deep_raw = seq_data[:, [1, 9, 10, 3, 5, 6]] # driver_id weekday slice_window distance link_num cross_num
        # self.deep_category = wide_deep_raw[:, :3] # driver_id slice_window
        # self.deep_real = wide_deep_raw[:, 3:]  # distance link_num cross_num
        # self.wide_index = wide_deep_raw.copy()  # [256, 5]
        # self.wide_index[:, 3:] = 0
        # self.wide_index += [0, drivers_num, drivers_num + 7, drivers_num + 7 + 288, drivers_num + 7 + 288 + 1, drivers_num + 7 + 288 + 1 + 1]  # WDR-LC 6个
        # self.wide_index = self.wide_index
        # self.wide_value = wide_deep_raw
        # self.wide_value[:, :3] = 1.0  # 类别特征的wide value 为1，只要其embedding后的值，连续特征的wide_value为连续值

    def __getitem__(self, index):
        attr = {}
        attr["departure"] = self.departure[index]
        attr["driver_id"] = self.driver_id[index]
        attr["weekday"] = self.weekday[index]
        attr['start_id'] = self.all_id[index][0]
        attr['end_id'] = self.all_id[index][-1]
        attr['mid_start_id'] = self.all_mid_start[index]
        attr["all_real"] = self.all_real[index]
        attr["all_real_re"] = self.all_real_re[index]
        attr["all_flow"] = self.all_flow[index]
        attr["all_flow_re"] = self.all_flow_re[index]
        attr["all_linkdistance"] = self.all_linkdistance[index]
        attr["all_linkdistance_re"] = self.all_linkdistance_re[index]
        attr["all_id"] = self.all_id[index]
        attr["all_id_re"] = self.all_id_re[index]
        attr["all_highway"] = self.all_highway[index]
        attr["all_highway_re"] = self.all_highway_re[index]
        attr["all_lane"] = self.all_lane[index]
        attr["all_lane_re"] = self.all_lane_re[index]
        attr["all_oneway"] = self.all_oneway[index]
        attr["all_oneway_re"] = self.all_oneway_re[index]
        attr["all_reversed"] = self.all_reversed[index]
        attr["all_reversed_re"] = self.all_reversed_re[index]
        attr["all_num"] = self.all_num[index]
        attr["all_mid_num"] = self.all_mid_num[index]
        attr["all_re_num"] = self.all_re_num[index]
        attr["targets"] = self.targets[index]
        attr["mid_targets"] = self.mid_targets[index]
        attr["re_targets"] = self.re_targets[index]
        return attr

    def __len__(self):
        return len(self.targets)


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # self.wide_index = data['wide_index']
        # self.wide_value = data['wide_value']
        # self.deep_category = data['deep_category']
        # self.deep_real = data['deep_real']
        self.departure = data['departure']
        self.driver_id = data['driver_id']
        self.weekday = data['weekday']
        self.start_id = data['start_id']
        self.end_id = data['end_id']
        self.all_link_feature = data['all_link_feature']
        self.all_re_num = data['all_re_num']
        self.all_flow = data['all_flow']
        self.all_linkdistance = data['all_linkdistance']
        self.all_real = data['all_real']
        self.mid_target = data['mid_target']
        self.re_target = data['re_target']
        self.mask = data['mask']

    def __getitem__(self, index):
        attr = {}
        attr["departure"] = self.departure[index]
        attr["driver_id"] = self.driver_id[index]
        attr["weekday"] = self.weekday[index]
        attr['start_id'] = self.start_id[index]
        attr['end_id'] = self.end_id[index]
        attr["all_link_feature"] = self.all_link_feature[index]
        attr["all_re_num"] = self.all_re_num[index]
        attr["all_flow"] = self.all_flow[index]
        attr["all_linkdistance"] = self.all_linkdistance[index]
        attr["all_real"] = self.all_real[index]
        attr["mid_target"] = self.mid_target[index]
        attr["re_target"] = self.re_target[index]
        attr["mask"] = self.mask[index]
        return attr

    def __len__(self):
        return len(self.re_target)

# 定义一个函数来处理不同的情况
def process_element(seqs, FLAGS_batch_size, max_num, dtype):
    element = np.zeros((FLAGS_batch_size, max_num), dtype=dtype)
    for i in range(FLAGS_batch_size):
        ele = seqs[i]
        element[i, 0:len(ele)] = np.array(ele) + 1
    return element

def process_and_pad_attributes(attrs, keys, data, batch_size, segment_num, float_keys, mask):
    for key in keys:
        element = process_element([item[key] for item in data], batch_size, segment_num,
                                  np.float32 if key in float_keys else np.int64)
        padded = torch.from_numpy(element).float() if key in float_keys else torch.from_numpy(element).long()
        attrs[key] = padded.unsqueeze(2)
    segment_mask = element > 0
    attrs[mask] = torch.from_numpy(segment_mask.astype(float)).float()
    attrs[mask] = attrs[mask].unsqueeze(2)

def collate_fn(data, FLAGS):
    nums = ['all_num', 'all_mid_num', 'all_re_num']
    ext_attrs = ['departure', 'driver_id', 'weekday']
    ods = ['start_id', 'end_id', 'mid_start_id']
    link_attrs = ['all_real', 'all_flow', 'all_linkdistance', 'all_highway', 'all_lane', 'all_oneway', 'all_reversed', 'all_id']
    er_link_attrs = ['all_real_re', 'all_flow_re', 'all_linkdistance_re',  'all_highway_re', 'all_lane_re', 'all_oneway_re', 'all_reversed_re', 'all_id_re']
    labels = ['targets', 'mid_targets', 're_targets']
    attrs = {}
    for key in nums:
        attrs[key] = torch.LongTensor([item[key] for item in data])
    for key in ext_attrs:
        attrs[key] = torch.LongTensor(np.array([item[key] for item in data]))
    for key in ods:
        attrs[key] = torch.LongTensor([item[key] for item in data])
    # 处理link_attrs中的键
    batch_size = len(data)
    # 处理link_attrs中的键
    process_and_pad_attributes(attrs, link_attrs, data, batch_size, FLAGS.segment_num,  ['all_real', 'all_flow', 'all_linkdistance'], 'mask')
    # 处理er_link_attrs中的键
    process_and_pad_attributes(attrs, er_link_attrs, data, batch_size, FLAGS.Lnum7, ['all_real_re', 'all_flow_re', 'all_linkdistance_re'], 'er_mask')
    for key in labels:
        attrs[key] = torch.tensor([item[key] for item in data], dtype=torch.int64)
    mask = attrs['mid_targets'] > 0
    for key in attrs:
        attrs[key] = attrs[key][mask]
    return attrs


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# 计算大小
def calculate_gap(target, predicts):
    gap = target - predicts
    return gap

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

