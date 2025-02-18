import numpy as np
import torch
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
    if mask is not None:
        segment_mask = element > 0
        attrs[mask] = torch.from_numpy(segment_mask.astype(float)).float()
        attrs[mask] = attrs[mask].unsqueeze(2)


def collate_fn(data, FLAGS):
    # 根据不同架构设置对应的变量名
    model_specific_attrs = {
        'Attention': {
            'nums': ['all_num', 'all_mid_num', 'all_re_num'],
            'ext_attrs': ['departure', 'driver_id', 'weekday'],
            'ods': ['start_id', 'end_id', 'mid_start_id'],
            'link_attrs': ['all_real', 'all_flow', 'all_linkdistance', 'all_highway', 'all_lane', 'all_oneway',
                           'all_reversed', 'all_id'],
            'er_link_attrs': ['all_real_re', 'all_flow_re', 'all_linkdistance_re', 'all_highway_re', 'all_lane_re',
                              'all_oneway_re', 'all_reversed_re', 'all_id_re']
        },
                           #     wide_index, wide_value, deep_category, deep_real, \
                           # all_num, all_mid_num, all_re_num, \
                           # all_id, all_real, all_flow, all_linkdistance, all_highway, all_lane, all_oneway, all_reversed, \
                           # all_id_re, all_real_re, all_flow_re, all_linkdistance_re, all_highway_re, all_lane_re, all_oneway_re, all_reversed_re, \
                           # targets, mid_targets, re_targets = [data[key].to(device) for key in data.keys()]
        'RNN': {
            'nums': ['all_num', 'all_mid_num', 'all_re_num'],
            'ext_attrs': ['wide_index', 'wide_value', 'deep_category', 'deep_real'],
            'ods': None,
            'link_attrs': ['all_id', 'all_real', 'all_flow', 'all_linkdistance', 'all_highway', 'all_lane',
                           'all_oneway', 'all_reversed'],
            'er_link_attrs': ['all_id_re', 'all_real_re', 'all_flow_re', 'all_linkdistance_re', 'all_highway_re',
                              'all_lane_re', 'all_oneway_re', 'all_reversed_re']
        },
        'MLP': {
            'nums': ['all_num', 'all_mid_num', 'all_re_num'],
            'ext_attrs': None,
            'ods': None,
            'link_attrs': ['all_id', 'all_real', 'all_flow', 'all_linkdistance', 'all_highway', 'all_lane',
                           'all_oneway', 'all_reversed'],
            'er_link_attrs': ['all_id_re', 'all_real_re', 'all_flow_re', 'all_linkdistance_re', 'all_highway_re',
                              'all_lane_re', 'all_oneway_re', 'all_reversed_re']
        }
    }

    # 根据model_type选择对应的配置
    if FLAGS.model == 'ConSTGAT' or FLAGS.model == 'SSML' or FLAGS.model == 'MetaER-TTE':
        config = model_specific_attrs['Attention']
    elif FLAGS.model == 'WDR_LC' or FLAGS.model == 'WDR':
        config = model_specific_attrs['RNN']
    elif FLAGS.model == 'MLPTTE':
        config = model_specific_attrs['MLP']
    else:
        raise ValueError('Unknown model type: {}'.format(FLAGS.model))
    nums = config['nums']
    ext_attrs = config.get('ext_attrs', [])
    ods = config.get('ods', [])
    link_attrs = config['link_attrs']
    er_link_attrs = config['er_link_attrs']
    labels = ['targets', 'mid_targets', 're_targets']

    attrs = {}

    # 处理数值类型
    for key in nums:
        attrs[key] = torch.LongTensor([item[key] for item in data])

    # 处理扩展属性（如果有）
    if ext_attrs:
        for key in ext_attrs:
            dtype = torch.LongTensor if key in ['wide_index', 'deep_category', 'departure', 'driver_id', 'weekday'] else torch.FloatTensor
            attrs[key] = dtype(np.array([item[key] for item in data], dtype=np.float32 if dtype == torch.FloatTensor else np.int64))

    # 处理ods属性（仅对ATT有效）
    if ods:
        for key in ods:
            attrs[key] = torch.LongTensor([item[key] for item in data])

    # 处理link_attrs中的键
    batch_size = len(data)
    process_and_pad_attributes(attrs, link_attrs, data, batch_size, FLAGS.segment_num,
                               ['all_real', 'all_flow', 'all_linkdistance'], 'mask' if FLAGS.model in ['ConSTGAT', 'SSML', 'MetaER-TTE'] else None)

    # 处理er_link_attrs中的键
    process_and_pad_attributes(attrs, er_link_attrs, data, batch_size, FLAGS.Lnum7,
                               ['all_real_re', 'all_flow_re', 'all_linkdistance_re'],
                               'er_mask' if FLAGS.model in ['ConSTGAT', 'SSML', 'MetaER-TTE'] else None)

    # 处理标签
    for key in labels:
        attrs[key] = torch.tensor([item[key] for item in data], dtype=torch.int64)

    # 处理mask
    mask = attrs['mid_targets'] > 0
    for key in attrs:
        attrs[key] = attrs[key][mask]

    return attrs


