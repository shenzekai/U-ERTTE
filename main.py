import os
import sys

import numpy as np
import torch
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
# torch.cuda.set_device(1)

from utills.log import logger_tb, message_logger

from FTML import *
from data_provider.Dataset import *
from data_provider.collect import *
# import nni

def train(model, optimizer, data_loader, loss):
    model.train()
    train_loss = []
    pr_losses = []
    er_losses = []
    for i, data in enumerate(data_loader):
        # Attention
        if FLAGS.model in ['ConSTGAT', 'SSML', 'MetaER-TTE']:
            all_num, all_mid_num, all_re_num, \
            departure, driver_id, weekday, start_id, end_id, mid_start_id, \
            all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, all_id, mask, \
            all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, all_id_re, re_mask, \
            targets, mid_targets, re_targets, = [data[key].to(device) for key in data.keys()]
            all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, L, 5]
            all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, L, 5]
            all_loss, pr_loss, er_loss = model(departure, driver_id, weekday, start_id, end_id, mid_start_id, \
                                               all_link_feature, all_real, all_flow, all_linkdistance, mask, \
                                               all_link_feature_re, all_real_re, all_flow_re, all_linkdistance_re, re_mask, \
                                               all_num, all_mid_num, all_re_num, \
                                               targets, mid_targets, re_targets, loss)
        # MLP
        elif FLAGS.model == 'MLPTTE':
            all_num, all_mid_num, all_re_num, \
            all_id, all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, \
            all_id_re, all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re ,\
            targets, mid_targets, re_targets = [data[key].to(device) for key in data.keys()]
            # print('id_range', max(all_id),min(all_id))
            all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(
                device)  # [B, F, 5]
            all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(
                device)  # [B, F, 5]
            all_loss, pr_loss, er_loss = model(all_link_feature, all_real, all_flow, all_linkdistance, \
                                               all_link_feature_re, all_real_re, all_flow_re, all_linkdistance_re, \
                                               all_num, all_mid_num, all_re_num, \
                                               targets, mid_targets, re_targets, loss)
        # WDR,WDR-LC
        elif FLAGS.model in ['WDR_LC', 'WDR']:
            all_num, all_mid_num, all_re_num, \
            wide_index, wide_value, deep_category, deep_real, \
            all_id, all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, \
            all_id_re, all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re ,\
            targets, mid_targets, re_targets = [data[key].to(device) for key in data.keys()]
            all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(
                device)  # [B, F, 5]
            all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(
                device)  # [B, F, 5]
            all_loss, pr_loss, er_loss = model(wide_index, wide_value, deep_category, deep_real, \
                                               all_link_feature, all_real, all_flow, all_linkdistance, \
                                               all_link_feature_re, all_real_re, all_flow_re, all_linkdistance_re, \
                                               all_num, all_mid_num, all_re_num, \
                                               targets, mid_targets, re_targets, loss)
        model.net.zero_grad()
        all_loss.backward()
        optimizer.step()
        train_loss.append(all_loss.item())
        pr_losses.append(pr_loss.item())
        er_losses.append(er_loss.item())

    train_loss = np.mean(np.array(train_loss), axis=0)
    pr_losses = np.mean(np.array(pr_losses), axis=0)
    er_losses = np.mean(np.array(er_losses), axis=0)
    print('train loss of in one epoch: ' + str(train_loss))
    print('pr loss of in one epoch: ' + str(pr_losses))
    print('er loss of in one epoch: ' + str(er_losses))


def train_process(batch_size, seq_data, model, optimizer, loss):
    dataset = TTEDataset(seq_data, FLAGS)
    data_loader = DataLoaderX(dataset, collate_fn=lambda x: collate_fn(x, FLAGS), batch_size=batch_size, num_workers=4)
    train(model, optimizer, data_loader, loss)


def val(model, val_data_loader, FLAGS):
    model.eval()
    predicts = []
    mid_predicts = []
    er_predicts = []
    er_targets = []
    pr_val_time = 0
    er_val_time = 0
    loss = FLAGS.loss
    label = []
    mid_label = []
    with torch.no_grad():
        for i, data in enumerate(val_data_loader):
            if FLAGS.model == 'SSML':
                all_num, all_mid_num, all_re_num, \
                departure, driver_id, weekday, start_id, end_id, mid_start_id, \
                all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, all_id, mask, \
                all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, all_id_re, re_mask, \
                targets, mid_targets, re_targets, = [data[key].to(device) for key in data.keys()]
                all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
                all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, F, 5]
                start_time = time.time()
                y, mid_y, mid_rep = model.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask, all_mid_num=all_mid_num)
                end_time = time.time()
                pr_val_time += end_time - start_time
                start_time = time.time()
                re_y = model.net(departure, driver_id, weekday, start_id, end_id, all_real_re, all_flow_re, all_linkdistance_re, all_link_feature_re, re_mask, all_re_num=all_re_num, mid_rep=mid_rep)
                end_time = time.time()
            elif FLAGS.model in ['ConSTGAT', 'MetaER-TTE']:
                all_num, all_mid_num, all_re_num, \
                departure, driver_id, weekday, start_id, end_id, mid_start_id, \
                all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, all_id, mask, \
                all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, all_id_re, re_mask, \
                targets, mid_targets, re_targets, = [data[key].to(device) for key in data.keys()]
                all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
                all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, F, 5]
                start_time = time.time()
                y, mid_y = model.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask, all_mid_num=all_mid_num)
                end_time = time.time()
                pr_val_time += end_time - start_time
                start_time = time.time()
                re_y = model.net(departure, driver_id, weekday, start_id, end_id, all_real_re, all_flow_re, all_linkdistance_re, all_link_feature_re, re_mask, all_re_num=all_re_num)
                end_time = time.time()
            elif FLAGS.model in ['WDR_LC', 'WDR']:
                all_num, all_mid_num, all_re_num, \
                wide_index, wide_value, deep_category, deep_real, \
                all_id, all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, \
                all_id_re, all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, \
                targets, mid_targets, re_targets, = [data[key].to(device) for key in data.keys()]
                all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
                all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, F, 5]
                start_time = time.time()
                y, mid_y = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature, all_num,all_flow,all_linkdistance, all_real, all_mid_num)
                end_time = time.time()
                pr_val_time += end_time - start_time
                start_time = time.time()
                re_y, re_target = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature_re, \
                                            all_re_num, all_flow_re, all_linkdistance_re, all_real_re,
                                            re_target=re_targets)
                end_time = time.time()
            elif FLAGS.model =='MLPTTE':
                all_num, all_mid_num, all_re_num, \
                all_id, all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, \
                all_id_re, all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, \
                targets, mid_targets, re_targets, = [data[key].to(device) for key in data.keys()]
                all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
                all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, F, 5]
                start_time = time.time()
                y, mid_y = model.net(all_link_feature, all_flow, all_linkdistance, all_real, all_num, all_mid_num)
                end_time = time.time()
                pr_val_time += end_time - start_time
                predicts += y.tolist()
                label += targets.tolist()
                mid_predicts += mid_y.tolist()
                mid_label += mid_targets.tolist()
                start_time = time.time()
                re_y = model.net(all_link_feature_re, all_flow_re, all_linkdistance_re, all_real_re, all_re_num)
                end_time = time.time()
            er_targets += re_targets.tolist()
            er_predicts += re_y.tolist()
            predicts += y.tolist()
            label += targets.tolist()
            mid_predicts += mid_y.tolist()
            mid_label += mid_targets.tolist()
            er_val_time += end_time - start_time
            er_targets += re_targets.tolist()
            er_predicts += re_y.tolist()

    print('PRTTE inference time: ' + str(pr_val_time))
    print('ERTTE inference time: ' + str(er_val_time))
    print('All inference time: ' + str(pr_val_time + er_val_time))
    throughput = calculate_throughput(len(val_data_loader.dataset), er_val_time)
    print(f"Val Throughput : {throughput:.2f} samples/second")
    predicts = np.array(predicts)
    label = np.array(label)
    mid_predicts = np.array(mid_predicts)
    mid_label = np.array(mid_label)
    er_predicts = np.array(er_predicts)
    er_targets = np.array(er_targets)
    metrics = calculate_all_metrics(loss, label, predicts, mid_label, mid_predicts, er_targets, er_predicts)
    return metrics['re_mape'], pr_val_time, er_val_time


def val_process(batch_size, seq_data, model, FLAGSe):
    val_dataset = TTEDataset(seq_data, FLAGS)
    val_data_loader = DataLoaderX(val_dataset, collate_fn=lambda x: collate_fn(x, FLAGS), batch_size=batch_size)
    mape, pr_val_time, er_val_time = val(model, val_data_loader, FLAGS)  # UQ
    return {'mape': mape, 'pr_val_time': pr_val_time, 'er_val_time': er_val_time}


def pre_test(model, test_data_loader, FLAGS, epoch):
    model.eval()
    # filtered dataset collect
    # re_y = model.net(departure, driver_id, weekday, mid_start_id, end_id, all_real_re, all_flow_re, all_linkdistance_re, all_link_feature_re, re_mask, all_re_num=all_re_num)
    mid_predicts = []
    er_targets_in = []
    er_predicts_in = []
    er_predicts = []
    er_targets = []
    departure_test = []
    driver_id_test = []
    weekday_test = []
    start_id_test = []
    end_id_test = []
    all_re_num_test = []
    all_flow_test = []
    all_real_test = []
    all_link_feature_test = []
    all_linkdistance_test = []
    mask_test = []
    mask_full_label, mask_full_predict, mask_mid_label, mask_mid_predict = [], [], [], []
    # 预测结果和label
    mid_target_test, re_target_test = [], []
    predicts, label = [], []
    mid_label = []
    if FLAGS.model in ['WDR_LC', 'WDR']:
        wide_index_test = []
        wide_value_test = []
        deep_category_test = []
        deep_real_test = []
    if FLAGS.model == 'SSML':
        mid_reps = []
    pr_test_time = 0
    er_test_time = 0
    loss = FLAGS.loss
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            if FLAGS.model == 'SSML':
                all_num, all_mid_num, all_re_num, \
                departure, driver_id, weekday, start_id, end_id, mid_start_id, \
                all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, all_id, mask, \
                all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, all_id_re, re_mask, \
                targets, mid_targets, re_targets = [data[key].to(device) for key in data.keys()]
                all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
                all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, F, 5]
                start_time = time.time()
                y, mid_y, mid_rep = model.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask, all_mid_num=all_mid_num)
                pr_test_time += time.time() - start_time
                start_time = time.time()
                re_y = model.net(departure, driver_id, weekday, mid_start_id, end_id, all_real_re, all_flow_re, all_linkdistance_re, all_link_feature_re, re_mask, all_re_num=all_re_num, mid_rep=mid_rep)
                end_time = time.time()
                er_test_time += end_time - start_time
                lower_bound = mid_y[:, 0]  # lower bound
                upper_bound = mid_y[:, 2]  # upper bound
                all_indices = torch.arange(all_num.size(0)).to(device)
                indice = torch.where((mid_targets < lower_bound) | (mid_targets > upper_bound))[0]  # can not filter indices
                # 计算补集索引
                mask = ~torch.isin(all_indices, indice)
                mask_full_label.append(targets[mask])
                mask_mid_label.append(mid_targets[mask])
                mask_full_predict.append(y[mask])
                mask_mid_predict.append(mid_y[mask])
                re_predict_in = y[mask] - mid_y[mask]
                er_predicts_in.append(re_predict_in if re_predict_in.ndim > 0 else re_predict_in.unsqueeze(0))
                re_targets_in = re_targets[mask]
                er_targets_in.append(re_targets_in if re_targets_in.ndim > 0 else re_targets_in.unsqueeze(0))
                departure_test += departure[indice]
                driver_id_test += driver_id[indice]
                weekday_test += weekday[indice]
                start_id_test += start_id[indice]
                end_id_test += end_id[indice]
                all_link_feature_test += all_link_feature_re[indice]
                all_re_num_test += all_re_num[indice]
                all_flow_test += all_flow_re[indice]
                all_linkdistance_test += all_linkdistance_re[indice]
                mask_test += re_mask[indice]
                all_real_test += all_real_re[indice]
                mid_reps += mid_rep[indice]
                mid_target_test += mid_targets[indice]
                re_target_test += re_targets[indice]
            elif FLAGS.model in ['ConSTGAT', 'MetaER-TTE']:
                all_num, all_mid_num, all_re_num, \
                departure, driver_id, weekday, start_id, end_id, mid_start_id, \
                all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, all_id, mask, \
                all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, all_id_re, re_mask, \
                targets, mid_targets, re_targets = [data[key].to(device) for key in data.keys()]
                all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
                all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, F, 5]
                start_time = time.time()
                y, mid_y = model.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow,all_linkdistance, all_link_feature, mask, all_mid_num=all_mid_num)
                pr_test_time += time.time() - start_time
                start_time = time.time()
                re_y = model.net(departure, driver_id, weekday, mid_start_id, end_id, all_real_re, all_flow_re,all_linkdistance_re, all_link_feature_re, re_mask, all_re_num=all_re_num)
                end_time = time.time()
                er_test_time += end_time - start_time
                lower_bound = mid_y[:, 0]  # lower bound
                upper_bound = mid_y[:, 2]  # upper bound
                # Create a tensor containing all indices
                all_indices = torch.arange(wide_index.size(0)).to(device)
                # Find indices where mid_targets are outside the specified bounds
                indice = torch.where((mid_targets < lower_bound) | (mid_targets > upper_bound))[0]
                # Compute the complement indices
                mask = ~torch.isin(all_indices, indice)
                mask_full_label.append(targets[mask])
                mask_mid_label.append(mid_targets[mask])
                mask_full_predict.append(y[mask])
                mask_mid_predict.append(mid_y[mask])
                re_predict_in = y[mask] - mid_y[mask]
                er_predicts_in.append(re_predict_in if re_predict_in.ndim > 0 else re_predict_in.unsqueeze(0))
                re_targets_in = re_targets[mask]
                er_targets_in.append(re_targets_in if re_targets_in.ndim > 0 else re_targets_in.unsqueeze(0))
                departure_test += departure[indice]
                driver_id_test += driver_id[indice]
                weekday_test += weekday[indice]
                start_id_test += start_id[indice]
                end_id_test += end_id[indice]
                all_link_feature_test += all_link_feature_re[indice]
                all_re_num_test += all_re_num[indice]
                all_flow_test += all_flow_re[indice]
                all_linkdistance_test += all_linkdistance_re[indice]
                mask_test += re_mask[indice]
                all_real_test += all_real_re[indice]
                mid_target_test += mid_targets[indice]
                re_target_test += re_targets[indice]
            elif FLAGS.model in ['WDR_LC', 'WDR']:
                all_num, all_mid_num, all_re_num, \
                wide_index, wide_value, deep_category, deep_real, \
                all_id, all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, \
                all_id_re, all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, \
                targets, mid_targets, re_targets, = [data[key].to(device) for key in data.keys()]
                all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
                all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, F, 5]
                start_time = time.time()
                y, mid_y = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature, all_num, all_flow, all_linkdistance, all_real, all_mid_num)
                pr_test_time += time.time() - start_time
                start_time = time.time()
                re_y, re_target = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature_re, \
                                            all_re_num, all_flow_re, all_linkdistance_re, all_real_re,
                                            re_target=re_targets)
                end_time = time.time()
                er_test_time += end_time - start_time
                lower_bound = mid_y[:, 0]  # upper bound
                upper_bound = mid_y[:, 2]  # lower bound
                # 创建一个包含所有索引的张量
                all_indices = torch.arange(wide_index.size(0)).to(device)
                indice = torch.where((mid_targets < lower_bound) | (mid_targets > upper_bound))[0]  # 不能过滤掉的
                # 计算补集索引
                mask = ~torch.isin(all_indices, indice)
                mask_full_label.append(targets[mask])
                mask_mid_label.append(mid_targets[mask])
                mask_full_predict.append(y[mask])
                mask_mid_predict.append(mid_y[mask])
                re_predict_in = y[mask] - mid_y[mask]
                er_predicts_in.append(re_predict_in if re_predict_in.ndim > 0 else re_predict_in.unsqueeze(0))
                re_targets_in = re_targets[mask]
                er_targets_in.append(re_targets_in if re_targets_in.ndim > 0 else re_targets_in.unsqueeze(0))
                wide_index_test += wide_index[indice]
                wide_value_test += wide_value[indice]
                deep_category_test += deep_category[indice]
                deep_real_test += deep_real[indice]
                all_link_feature_test += all_link_feature_re[indice]
                all_re_num_test += all_re_num[indice]
                all_flow_test += all_flow_re[indice]
                all_linkdistance_test += all_linkdistance_re[indice]
                all_real_test += all_real_re[indice]
                mid_target_test += mid_targets[indice]
                re_target_test += re_targets[indice]
            elif FLAGS.model == 'MLPTTE':
                all_num, all_mid_num, all_re_num, \
                all_id, all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, \
                all_id_re, all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, \
                targets, mid_targets, re_targets, = [data[key].to(device) for key in data.keys()]
                all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
                all_link_feature_re = torch.cat([all_id_re, all_highway_re, all_lane_re, all_reversed_re, all_oneway_re], dim=2).to(device)  # [B, F, 5]
                start_time = time.time()
                y, mid_y = model.net(all_link_feature, all_flow, all_linkdistance, all_real, all_num, all_mid_num)
                pr_test_time += time.time() - start_time
                start_time = time.time()
                re_y = model.net(all_link_feature_re, all_flow_re, all_linkdistance_re, all_real_re, all_re_num)
                end_time = time.time()
                er_test_time += end_time - start_time
                # UGD mechanism
                lower_bound = mid_y[:, 0]  # 上界
                upper_bound = mid_y[:, 2]  # 下界
                # 创建一个包含所有索引的张量
                all_indices = torch.arange(all_num.size(0)).to(device)
                indice = torch.where((mid_targets < lower_bound) | (mid_targets > upper_bound))[0]  # 不能过滤掉的
                # 计算补集索引
                mask = ~torch.isin(all_indices, indice)
                mask_full_label.append(targets[mask])
                mask_mid_label.append(mid_targets[mask])
                mask_full_predict.append(y[mask])
                mask_mid_predict.append(mid_y[mask])
                re_predict_in = y[mask] - mid_y[mask]
                er_predicts_in.append(re_predict_in if re_predict_in.ndim > 0 else re_predict_in.unsqueeze(0))
                re_targets_in = re_targets[mask]
                er_targets_in.append(re_targets_in if re_targets_in.ndim > 0 else re_targets_in.unsqueeze(0))
                all_link_feature_test += all_link_feature_re[indice]
                all_re_num_test += all_re_num[indice]
                all_flow_test += all_flow_re[indice]
                all_linkdistance_test += all_linkdistance_re[indice]
                all_real_test += all_real_re[indice]
                mid_target_test += mid_targets[indice]
                re_target_test += re_targets[indice]
            else:
                raise ValueError('model name error')
            predicts.append(y)
            label.append(targets)
            mid_predicts.append(mid_y)
            mid_label.append(mid_targets)
            er_targets.append(re_targets)
            er_predicts.append(re_y)
    print('PRTTE inference time: ' + str(pr_test_time))
    print('ERTTE inference time: ' + str(er_test_time))
    print('All inference time: ' + str(pr_test_time + er_test_time))
    label = torch.cat(label)
    predicts = torch.cat(predicts)
    mid_label = torch.cat(mid_label)
    mid_predicts = torch.cat(mid_predicts)
    er_targets_in = torch.cat(er_targets_in)
    er_predicts_in = torch.cat(er_predicts_in)
    metrics = calculate_all_metrics(loss, label.cpu().numpy(), predicts.cpu().numpy(), mid_label.cpu().numpy(), \
                                    mid_predicts.cpu().numpy(), er_targets_in.cpu().numpy(),
                                    er_predicts_in.cpu().numpy(), is_in=True)
    er_targets = torch.cat(er_targets)
    er_predicts = torch.cat(er_predicts)
    full_metrics=calculate_metrics(er_targets.cpu().numpy(), er_predicts.cpu().numpy(), loss)
    print("ERTTE Full Test Result:", len(label))
    print('MAPE:%.3f\tMAE:%.3f\tRMSE:%.3f\tSR:%.3f' % (full_metrics['mape'] * 100, full_metrics['mae'], full_metrics['rmse'], full_metrics['sr']))
    print('PICP:%.3f\tMPIW:%.3f\tMIS:%.3f' % (full_metrics['picp'], full_metrics['mpiw'], full_metrics['mis']))
    FilterData = {}
    FilterData['all_link_feature'] = torch.stack(all_link_feature_test)
    FilterData['all_re_num'] = torch.stack(all_re_num_test)
    FilterData['all_flow'] = torch.stack(all_flow_test)
    FilterData['all_linkdistance'] = torch.stack(all_linkdistance_test)
    FilterData['all_real'] = torch.stack(all_real_test)
    FilterData['mid_target'] = torch.stack(mid_target_test)
    FilterData['re_target'] = torch.stack(re_target_test)
    if FLAGS.model in ['MetaER-TTE', 'ConSTGAT', 'SSML']:
        FilterData['departure'] = departure_test
        FilterData['driver_id'] = driver_id_test
        FilterData['weekday'] = weekday_test
        FilterData['start_id'] = start_id_test
        FilterData['end_id'] = end_id_test
        if FLAGS.model == 'SSML':
            FilterData['mid_rep'] = torch.stack(mid_reps)
        FilterData['mask'] = torch.stack(mask_test)
    elif FLAGS.model in ['WDR_LC', 'WDR']:
        FilterData['wide_index'] = torch.stack(wide_index_test)
        FilterData['wide_value'] = torch.stack(wide_value_test)
        FilterData['deep_category'] = torch.stack(deep_category_test)
        FilterData['deep_real'] = torch.stack(deep_real_test)

    throughput = calculate_throughput(len(test_data_loader.dataset) + len(er_predicts_in), er_test_time)
    print(f"Test Throughput : {throughput:.2f} samples/second")
    return FilterData, pr_test_time, er_targets_in, er_predicts_in


def test(model, test_data_loader, FLAGS, epoch, er_targets_in, er_predicts_in):
    predicts = []
    label = []
    loss = FLAGS.loss
    er_test_time = 0
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            if FLAGS.model == 'SSML':
                departure, driver_id, weekday, start_id, end_id, mid_rep, all_link_feature, all_re_num, all_flow, all_linkdistance, all_real, mid_target, re_target, mask = [
                data[k].to(device) for k in data]
                start_time = time.time()
                y = model.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask, all_re_num=all_re_num, mid_rep=mid_rep)
                end_time = time.time()
            elif FLAGS.model in ['MetaER-TTE', 'ConSTGAT']:
                departure, driver_id, weekday, start_id, end_id, all_link_feature, all_re_num, all_flow, all_linkdistance, all_real, mid_target, re_target, mask = [
                    data[k].to(device) for k in data]
                start_time = time.time()
                y = model.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance,  all_link_feature, mask, all_re_num=all_re_num)
                end_time = time.time()
            elif FLAGS.model in ['WDR_LC', 'WDR']:
                wide_index, wide_value, deep_category, deep_real, all_link_feature, all_re_num, all_flow, all_linkdistance, \
                    all_real, mid_target, re_target = [data[k].to(device) for k in data]
                start_time = time.time()
                y, target = model.net(wide_index, wide_value, deep_category, deep_real, all_link_feature, \
                                      all_re_num, all_flow, all_linkdistance, all_real, mid_target, \
                                      re_target=re_target)
                end_time = time.time()
            elif FLAGS.model == 'MLPTTE':
                all_link_feature, all_re_num, all_flow, all_linkdistance, all_real, mid_target, re_target = [data[k].to(device) for k in data]
                start_time = time.time()
                y = model.net(all_link_feature, all_flow, all_linkdistance, all_real, all_re_num)
                end_time = time.time()
            er_test_time += end_time - start_time
            predicts += y.tolist()
            label += re_target.tolist()
    print('Test ERTTE inference time: ' + str(er_test_time))
    predicts = np.array(predicts)
    label = np.array(label)
    metrics = calculate_metrics(label, predicts, loss)
    print("Need ER-TTE RESULTS:", len(label), predicts.shape)
    print('MAPE:%.3f\tMAE:%.3f\tRMSE:%.3f\tSR:%.3f' % (
    metrics['mape'] * 100, metrics['mae'], metrics['rmse'], metrics['sr']))
    print('PICP:%.3f\tMPIW:%.3f\tMIS:%.3f' % (metrics['picp'], metrics['mpiw'], metrics['mis']))
    label = np.concatenate((label, er_targets_in.cpu().numpy()), axis=0)
    predicts = np.concatenate((predicts, er_predicts_in.cpu().numpy()), axis=0)
    avg_metrics = calculate_metrics(label, predicts, loss)
    print("AVG ER-TTE RESULTS:")
    print('MAPE:%.3f\tMAE:%.3f\tRMSE:%.3f\tSR:%.3f' % (
    avg_metrics['mape'] * 100, avg_metrics['mae'], avg_metrics['rmse'], avg_metrics['sr']))
    print('PICP:%.3f\tMPIW:%.3f\tMIS:%.3f' % (avg_metrics['picp'], avg_metrics['mpiw'], avg_metrics['mis']))
    return avg_metrics['mape'], er_test_time


def process_test(batch_size, seq_data, model, FLAGS, epoch, is_test=False):
    pre_test_dataset = TTEDataset(seq_data, FLAGS)
    pre_test_dataloader = DataLoaderX(pre_test_dataset, collate_fn=lambda x: collate_fn(x, FLAGS),
                                      batch_size=batch_size, num_workers=4)
    start_time = time.time()
    data, pr_test_time, er_targets_in, er_predicts_in = pre_test(model, pre_test_dataloader, FLAGS, epoch)  # UQ
    end_time = time.time()
    print("Pre-test time: " + str(end_time - start_time))
    test_dataset = TestDataset(data,FLAGS)
    test_dataloder = DataLoaderX(test_dataset, batch_size=batch_size)
    start_time = time.time()
    mape, er_test_time = test(model, test_dataloder, FLAGS, epoch, er_targets_in, er_predicts_in)
    end_time = time.time()
    print("Filter test time: " + str(end_time - start_time))
    return {'mape': mape, 'pr_test_time': pr_test_time, 'er_test_time': er_test_time}


def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file


def main(ws, epochs, FLAGS, is_test=True):
    batch_size = FLAGS.batch_size
    learning_rate = 1e-3
    total_train_time = 0
    total_val_time = 0
    total_test_time = 0
    patience = 0
    # tuner_params = nni.get_next_parameter()  # 这会获得一组搜索空间中的参数
    # params.update(tuner_params)

    model = FTML(FLAGS).to(device)
    optimizer = torch.optim.Adam(params=model.net.parameters(), lr=learning_rate, weight_decay=1e-3)
    lr_decay_step = 10 if not is_test else 1
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.9)
    if FLAGS.loss == 'maemis':
        loss = maemis_loss
    elif FLAGS.loss == 'quantile':
        loss = quantile_loss
    elif FLAGS.loss == 'MAE':
        loss = nn.L1Loss()
    else:
        loss = mape
    print("LOSS:", loss)
    filepath = ws
    # test_seq_name = filepath + 'test.npy'

    # train process
    # ncf_scores = []
    best = 100
    best_epoch = 0
    train_data = np.load(filepath + 'train.npy', allow_pickle=True)
    val_data = np.load(filepath + '/val.npy', allow_pickle=True)
    test_data = np.load(filepath + 'test.npy', allow_pickle=True)
    train_data = train_data[:int(train_data.shape[0]*FLAGS.scale)]
    train_data[:, 10] = train_data[:, 10] - 1
    val_data[:, 10] = val_data[:, 10] - 1
    test_data[:, 10] = test_data[:, 10] - 1
    for i in range(epochs):
        print('#' * 50)
        print('num epoch: ' + str(i))
        print('training...')
        print(f'current learning rate: {optimizer.param_groups[0]["lr"]}')
        start_time = time.time()
        train_process(batch_size, train_data, model, optimizer, loss)
        end_time = time.time()
        epoch_train_time = end_time - start_time
        print('train time: ' + str(epoch_train_time))
        total_train_time += epoch_train_time
        StepLR.step()
        if is_test:
            # 验证集计算
            print('validating...')
            start_time = time.time()
            val_eval = val_process(batch_size, val_data, model, FLAGS)
            end_time = time.time()
            print('All val time: ' + str(end_time - start_time))
            val_eval['er_val_time'] = end_time - start_time
            if val_eval['mape'] <= best:
                best = val_eval['mape']
                best_epoch = i
                print('best mape: %.3f' % best)
                patience = 0
            else:
                patience += 1
                print('patience: %d' % patience)
            if patience >= FLAGS.patience:
                print('early stop in epoch %d' % i)
                print('best mape: %.3f in epoch %d' % (best, best_epoch))
                break
            total_val_time += val_eval['er_val_time']
            # nni.report_intermediate_result(val_eval['mape'])
            # 保存模型
            model_path = output_path + f'{model_name}_val-mape{(val_eval["mape"] * 100):.3f}-epoch{i}.pth'
            check_point = {
                "model": model.state_dict(),
                "epoch": i,
            }
            torch.save(check_point, model_path)
            # 输出测试集
            print('testing...')
            start_time = time.time()
            test_eval = process_test(batch_size, test_data, model, FLAGS, i, is_test=True)
            end_time = time.time()
            print('epoch Full test time: ' + str(end_time - start_time))
            total_test_time += test_eval['er_test_time']
        if is_test and i == 30:
            return
    average_train_time = total_train_time / i
    average_val_time = total_val_time / i
    average_test_time = total_test_time / i
    print(f"Average training time per epoch: {average_train_time:.3f} seconds")
    print(f"Average validation time per epoch: {average_val_time:.3f} seconds")
    print(f"Average testing time per epoch: {average_test_time:.3f} seconds")
    # nni.report_final_result(val_eval['mape'])


parser = argparse.ArgumentParser(description='Entry Point of the code')
# Model parameters
parser.add_argument('--loss', type=str, default='quantile', help="loss function")
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=4096, help="batch size")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--model', type=str, default='SSML')
parser.add_argument('--hidden_dim', type=float, default=64, help="hidden dimension")
parser.add_argument('--update_step', type=int, default=5, help="update step")
parser.add_argument('--isdropout', type=bool, default=False, help="MC-dropout")
parser.add_argument('--is_filter', type=bool, default=False, help="filter")
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--er_mode', type=int, default=0, help='test')
# dataset parameters
parser.add_argument('--drivers_num', type=int, default=56549, help="number of drivers") # 56549 438
parser.add_argument('--num_components', type=int, default=6779, help="number of Table components") # 6779 5282
parser.add_argument('--segment_num', type=int, default=120, help="segment number per link") # 192
parser.add_argument('--Lnum7', type=int, default=83, help="0.7 remain segment") #
parser.add_argument('--Lnum3', type=int, default=36, help="0.3 pre segment") # 37 36
parser.add_argument('--highway_num', type=int, default=21, help="highway categories") # XIAN 21 Porto 20
parser.add_argument('--lane_num', type=int, default=13, help="lane categories")
parser.add_argument('--oneway', type=int, default=2, help="oneway categories")
parser.add_argument('--reversed', type=int, default=3, help="reversed")
parser.add_argument('--scale', type=float, default=0.2, help="scale")
# backup
parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--code_backup', type=bool, default=True, help='code backup or not')
parser.add_argument('--path', type=str, default='/mnt/nfsData10/data/XAData/4_300_3000_7200/', help='data path')
FLAGS = parser.parse_args()
logger = logger_tb(FLAGS.log_dir, FLAGS.model, FLAGS.code_backup)
sys.stdout = message_logger(logger.log_dir)

if __name__ == "__main__":
    # params = vars(get_params())
    print(FLAGS)
    epochs = FLAGS.epochs
    # ws = get_workspace()
    ws = FLAGS.path
    model_name = FLAGS.model
    is_test = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    # train three times
    for i in range(3):
        output_path = ws + f'/results/trained_model_{i}_{model_name}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'current round: {i}')
        main(ws, epochs, FLAGS, is_test)