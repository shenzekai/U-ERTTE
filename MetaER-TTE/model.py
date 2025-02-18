from utills import *
import numpy as np
import torch
import torch.nn as nn
import math

# layer部分
class LinkFeatureEmbedding(nn.Module):
    def __init__(self, feature_ranges, embedding_dims):
        super(LinkFeatureEmbedding, self).__init__()
        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=range + 1, embedding_dim=dim)
            for range, dim in zip(feature_ranges, embedding_dims)
        ])

        self.initialize_embeddings()

    def initialize_embeddings(self):
        for embedding in self.feature_embeddings:
            # 根据嵌入的维度来设置初始化范围
            # dim = 14 / embedding.embedding_dim
            # 使用均匀分布初始化
            # nn.init.uniform_(embedding.weight, -1 / dim, 1 / dim)
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # 将所有嵌入的输出在最后一个维度上拼接起来
        embedded_features = [embedding(x[:, :, i]) for i, embedding in enumerate(self.feature_embeddings)]
        return torch.cat(embedded_features, dim=-1)

# 基础模型 ConSTGAT
class ConstGAT(nn.Module):
    def __init__(self,  FLAGS):
        super(ConstGAT, self).__init__()
        # 特征的范围
        # all_link_feature = torch.cat([all_id, all_highway, all_lane, all_reversed, all_oneway], dim=2).to(device)  # [B, F, 5]
        feature_ranges = [FLAGS.num_components, FLAGS.highway_num, FLAGS.lane_num, FLAGS.reversed, FLAGS.oneway]
        embedding_dims = [16, 5, 4, 2, 2] # 29
        self.hidden_dim = FLAGS.hidden_dim
        self.attention = Attention(self.hidden_dim)
        # self.LinkEmbeddings = nn.Embedding(FLAGS.num_components, 16)
        self.feature_embeddings = LinkFeatureEmbedding(feature_ranges, embedding_dims)
        self.startEmbeddings = nn.Embedding(FLAGS.num_components, 16)
        self.endEmbeddings = nn.Embedding(FLAGS.num_components, 16)
        self.driverEmbeddings = nn.Embedding(FLAGS.drivers_num + 1, 20)
        self.TimeEmbeddings = nn.Embedding(288, 10)
        self.weekdayEmbeddings = nn.Embedding(7, 3)

    def forward(self, departure, driver_id, weekday, start_id, end_id,  all_real, all_flow, all_linkdistance, all_link_feature, mask, segment_num):
        start_id_emb = self.startEmbeddings(start_id)
        start_id_emb = start_id_emb.unsqueeze(1).repeat(1, segment_num, 1)
        end_id_emb = self.endEmbeddings(end_id)
        end_id_emb = end_id_emb.unsqueeze(1).repeat(1, segment_num, 1)
        all_feature_emb = self.feature_embeddings(all_link_feature)
        departure_emb = self.TimeEmbeddings(departure)
        departure_emb = departure_emb.unsqueeze(1).repeat(1, segment_num, 1)
        driver_emb = self.driverEmbeddings(driver_id)
        driver_emb = driver_emb.unsqueeze(1).repeat(1, segment_num, 1)
        weekday_emb = self.weekdayEmbeddings(weekday)
        weekday_emb = weekday_emb.unsqueeze(1).repeat(1, segment_num, 1)

        # mid_targets = mid_targets.unsqueeze(2).repeat(1, segment_num, 1)
        query_feature = torch.cat([all_feature_emb, driver_emb, start_id_emb, end_id_emb, departure_emb, weekday_emb,\
                                   all_real, all_flow], dim=2)  # 29+10+16+16+20+3+2=96
        # query_feature_reshape = query_feature.view(-1, 1, query_feature.shape[-1])
        '''
        departure = departure.repeat_interleave(
                repeats=all_id.size(0), dim=0)
        '''

        # key_feature = torch.cat([all_id_emb, departure_emb, all_flow], dim=2)
        key_feature = query_feature
        value_feature = key_feature

        attention = self.attention(query_feature, key_feature, value_feature, mask)
        # att_link = torch.sum(att_dist_link.unsqueeze(dim=-1) * link_context_feat, [1])
        out_features = torch.cat([query_feature, attention], dim=2)
        # out_features = attention
        return out_features


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(96, self.hidden_dim)
        self.key = nn.Linear(96, self.hidden_dim)
        self.value = nn.Linear(96, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # self.layer_norm=nn.LayerNorm(self.hidden_dim)
    def forward(self, query, key, value, mask):
        query = self.query(query)  # B,L,hidden_dim
        key = self.key(key)
        value = self.value(value)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_dim)  # B,L,L
        if mask is not None:
            score = score.masked_fill_(mask == 0, -1e10)
        attention = F.softmax(score, dim=-1)  # B,L,L
        attention = torch.matmul(attention, value)
        attention = self.layer_norm(attention)
        return attention

class TaskCluster(nn.Module):
    def __init__(self, FLAGS):
        super(TaskCluster, self).__init__()
        self.ClusterNum = 3
        self.ClusterCenters = nn.Parameter(torch.randn(self.ClusterNum, FLAGS.hidden_dim+96))

    def forward(self, query):
        score = torch.matmul(query, self.ClusterCenters.T)
        score = F.softmax(score, dim=-1)  # [B,3]
        X_CE = torch.einsum('bk,bf->bf', score, query)  # [B,F]
        return score, X_CE

# 聚类感知参数存储器
class ClusterAwareParameterMemory(nn.Module):
    def __init__(self, num_clusters, feature_dim, dest_dim):
        super(ClusterAwareParameterMemory, self).__init__()
        self.alpha = 0.3
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.dest_dim = dest_dim
        self.cluster_memory = nn.Parameter(torch.randn(num_clusters, feature_dim, dest_dim))

    def forward(self, cluster_assignments):
        self.cluster_memory = (1-self.alpha)*self.cluster_memory + self.alpha*torch.matmul(cluster_assignments.unsqueeze(2), self.cluster_memory).squeeze(2)
        return torch.matmul(cluster_assignments.unsqueeze(2), self.cluster_memory).squeeze(2)


class LinkFeatureEmbedding(nn.Module):
    def __init__(self, feature_ranges, embedding_dims):
        super(LinkFeatureEmbedding, self).__init__()
        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=range + 1, embedding_dim=dim)
            for range, dim in zip(feature_ranges, embedding_dims)
        ])

        self.initialize_embeddings()

    def initialize_embeddings(self):
        for embedding in self.feature_embeddings:
            # 根据嵌入的维度来设置初始化范围
            dim = 14 / embedding.embedding_dim
            # 使用均匀分布初始化
            nn.init.uniform_(embedding.weight, -1 / dim, 1 / dim)
            # nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # 将所有嵌入的输出在最后一个维度上拼接起来
        embedded_features = [embedding(x[:, :, i]) for i, embedding in enumerate(self.feature_embeddings)]
        return torch.cat(embedded_features, dim=-1)

# model部分
class ConstGATModel(nn.Module):
    def __init__(self, FLAGS):
        super(ConstGATModel, self).__init__()
        self.constgat = ConstGAT(FLAGS)
        self.TaskCluster = TaskCluster(FLAGS)
        self.alpha = 0.7
        # self.MLP = nn.Linear((FLAGS.hidden_dim+78)*2, 1)
        self.MLP = nn.Linear((FLAGS.hidden_dim + 96)*2, 3)
    def forward(self, departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask,  all_mid_num=None, all_re_num=None):
        segment_num = all_link_feature.size(1)
        x = self.constgat(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask, segment_num)
        full_rep = torch.sum(x, dim=1)
        if all_mid_num is not None:
            mask = torch.arange(x.size(1)).unsqueeze(0).to(device) < all_mid_num.unsqueeze(1)
            mid_rep = x * mask.unsqueeze(2).float()
            mid_rep = torch.sum(mid_rep, dim=1)
            mid_cluster_score, mid_X_CE = self.TaskCluster(mid_rep)
            mid_rep = torch.cat([mid_rep, mid_X_CE], dim=1)
            max_score = torch.max(mid_cluster_score, dim=1, keepdim=True)[0]
            self.MLP.weight.data = update_weights(max_score, self.MLP.weight.data, self.alpha)
            mid_results = self.MLP(mid_rep)
            ClusterScore, X_CE = self.TaskCluster(full_rep)
            full_rep = torch.cat([full_rep, X_CE], dim=1)
            max_score = torch.max(ClusterScore, dim=1, keepdim=True)[0]
            self.MLP.weight.data = update_weights(max_score, self.MLP.weight.data, self.alpha)
            pred = self.MLP(full_rep)
            return pred, mid_results
        if all_re_num is not None:
            ClusterScore, X_CE = self.TaskCluster(full_rep)
            full_rep = torch.cat([full_rep, X_CE], dim=1)
            max_score = torch.max(ClusterScore, dim=1, keepdim=True)[0]
            self.MLP.weight.data = update_weights(max_score, self.MLP.weight.data, self.alpha)
            pred = self.MLP(full_rep)
        return pred


def update_weights(max_score, weights, alpha):
    for i in range(3):
        NewWeight = max_score * weights[i:i+1]
        NewWeight = torch.mean(NewWeight, dim=0, keepdim=True)
        weights[i:i+1] = alpha * weights[i:i+1] + (1 - alpha) * NewWeight
    return weights

def enable_dropout(m):
    print("MCDropout enabled")
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()