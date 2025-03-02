
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Concatenate all embedding outputs along the last dimension
        embedded_features = [embedding(x[:, :, i]) for i, embedding in enumerate(self.feature_embeddings)]
        return torch.cat(embedded_features, dim=-1)

class ConstGAT(nn.Module):
    def __init__(self,  FLAGS):
        super(ConstGAT, self).__init__()
        feature_ranges = [FLAGS.num_components, FLAGS.highway_num, FLAGS.lane_num, FLAGS.reversed, FLAGS.oneway]
        embedding_dims = [16, 5, 4, 2, 2] # 29
        self.hidden_dim = FLAGS.hidden_dim
        self.attention = Attention(96, self.hidden_dim)
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
        out_features = torch.cat([query_feature, attention], dim=2)  # 96+64=160
        # out_features = attention
        return out_features


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, self.hidden_dim)
        self.key = nn.Linear(input_dim, self.hidden_dim)
        self.value = nn.Linear(input_dim, self.hidden_dim)
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

# model
class ConstGATModel(nn.Module):
    def __init__(self, FLAGS):
        super(ConstGATModel, self).__init__()
        self.constgat = ConstGAT(FLAGS)
        self.attention = Attention(160, FLAGS.hidden_dim)
        # self.alpha = 0.7
        self.FullMLP = nn.Linear(FLAGS.hidden_dim + 96, 3)
        self.ReMLP = nn.Linear(FLAGS.hidden_dim + 160, 3)
    def forward(self, departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask,  all_mid_num=None, all_re_num=None, mid_rep=None):
        segment_num = all_link_feature.size(1)
        x = self.constgat(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask, segment_num)
        full_rep = torch.sum(x, dim=1)
        if all_mid_num is not None:
            mask = torch.arange(x.size(1)).unsqueeze(0).to(device) < all_mid_num.unsqueeze(1)
            mid_rep = x * mask.unsqueeze(2).float()
            mid_rep_sum = torch.sum(mid_rep, dim=1)
            mid_results = self.FullMLP(mid_rep_sum)
            pred = self.FullMLP(full_rep)
            return pred, mid_results, mid_rep
        if all_re_num is not None:
            mid_rep = mid_rep[:, :x.shape[1]]
            att_score = self.attention(mid_rep, x, x, mask)
            x = torch.cat([att_score, x], dim=-1)
            x = torch.sum(x, dim=1)
            pred = self.ReMLP(x)
        return pred


