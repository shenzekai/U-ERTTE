import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeaturesLinear(nn.Module):
    def __init__(self, field_dims, output_dim):
        """
        :param field_dims: Total number of unique features across all fields
        :param output_dim: Dimension of the output embedding
        """
        super(FeaturesLinear, self).__init__()
        self.fc = nn.Embedding(field_dims, output_dim)  # Embedding layer for linear transformation
        self.bias = nn.Parameter(torch.zeros((output_dim,)))  # Learnable bias term

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Sum feature embeddings along fields and add bias
        return torch.sum(self.fc(x), dim=1) + self.bias

class DeepFeaturesEmbedding(nn.Module):

    def __init__(self, deep_feature_ranges, deep_embed_dims):
        super(DeepFeaturesEmbedding, self).__init__()
        self.DeepEmbedding = nn.ModuleList([nn.Embedding(num_embeddings=range + 1, embedding_dim=dim) for range, dim in
                                            zip(deep_feature_ranges, deep_embed_dims)])
        for embedding in self.DeepEmbedding:
            nn.init.xavier_normal_(embedding.weight.data)

    def forward(self, x):
        """
            :param x: Long tensor of size ``(batch_size, num_fields)``
            """
        embedded_features = [embedding(x[:, i]) for i, embedding in enumerate(self.DeepEmbedding)]
        # Concatenate all embedding outputs along the last dimension
        return torch.cat(embedded_features, dim=1)


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = nn.Embedding(field_dims, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)


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


class FactorizationMachine(nn.Module):

    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        return 0.5 * ix


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim, mid=True):
        super(Regressor, self).__init__()
        self.linear_wide = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_deep = nn.Linear(input_dim, output_dim, bias=False)
        if mid:
            self.linear_recurrent = nn.Linear(input_dim * 2, output_dim, bias=False)
        else:
            self.linear_recurrent = nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer = MultiLayerPerceptron(output_dim, (output_dim,), dropout=0.15, output_layer=True)

    def forward(self, wide, deep, recurrent):
        fuse = self.linear_wide(wide) + self.linear_deep(deep) + self.linear_recurrent(recurrent)
        return self.out_layer(fuse)


# model部分
class WDR(nn.Module):

    def __init__(self, FLAGS):
        super(WDR, self).__init__()

        drivers_num = FLAGS.drivers_num
        wide_field_dims = np.array([drivers_num, 7, 288, 1, 1, 1])
        # wide_field_dims_re = np.array([drivers_num, 288, 1, 1, 1, 1])
        wide_embed_dim = 20
        wide_mlp_dims = (256,)
        # deep_field_dims = np.array([drivers_num, 288])
        deep_feature_ranges = [drivers_num, 7, 288]
        deep_embed_dims = [20, 4, 20]

        # deep_real_dim = 3  # WDR-LC
        deep_real_dim = 2  # WDR
        # deep_category_dim = 3
        # deep_category_dim = 3

        deep_mlp_input_dim = sum(deep_embed_dims) + deep_real_dim
        deep_mlp_dims = (256,)
        # id_dims = FLAGS.num_components
        # id_embed_dim = 20
        feature_ranges = [FLAGS.num_components, FLAGS.highway_num, FLAGS.lane_num, FLAGS.reversed, FLAGS.oneway]
        embedding_dims = [16, 5, 4, 2, 2]
        all_embed_dim = 29
        mlp_out_dim = 256
        gru_hidden_size = 256
        reg_input_dim = 256
        reg_output_dim = 256

        self.wide_embedding = FeaturesEmbedding(sum(wide_field_dims), wide_embed_dim)
        self.fm = nn.Sequential(
            FactorizationMachine(),
            nn.BatchNorm1d(wide_embed_dim),
        )
        self.wide_mlp = MultiLayerPerceptron(wide_embed_dim, wide_mlp_dims)  # 不batchnorm
        self.feature_embeddings = LinkFeatureEmbedding(feature_ranges, embedding_dims)
        self.deep_feature_embeddings = DeepFeaturesEmbedding(deep_feature_ranges, deep_embed_dims)
        self.deep_mlp = MultiLayerPerceptron(deep_mlp_input_dim, deep_mlp_dims, dropout=0.15)
        self.all_mlp = nn.Sequential(
            nn.Linear(all_embed_dim + 3, mlp_out_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=mlp_out_dim, hidden_size=gru_hidden_size, num_layers=2, batch_first=True)
        self.regressor = Regressor(reg_input_dim, reg_output_dim, mid=False)

    def forward(self, wide_index, wide_value, deep_category, deep_real,
                all_link_feature, all_num, all_flow, all_linkdistance, all_real, mid_num=None,
                re_target=None):
        self.flatten_parameters()
        batch_size = wide_index.size(0)
        # deep part
        deep_embedding = self.deep_feature_embeddings(deep_category)
        # wide
        wide_embedding = self.wide_embedding(wide_index)  # Embed all item features; continuous features get 1D embedding
        cross_term = self.fm(wide_embedding * wide_value.unsqueeze(2))  # Compute cross terms; wide_value contains 1s and dense feature values
        # deep_input = torch.cat([deep_embedding, deep_real, mid_targets.unsqueeze(1)], dim=1)  # Concatenate deep features
        deep_input = torch.cat([deep_embedding, deep_real], dim=1)
        deep_output = self.deep_mlp(deep_input)
        wide_output = self.wide_mlp(cross_term)
        # recurrent part
        all_feature_embedding = self.feature_embeddings(all_link_feature)
        all_input = torch.cat([all_feature_embedding, all_real, all_flow, all_linkdistance], dim=2)
        recurrent_input = self.all_mlp(all_input)
        packed_all_input = pack_padded_sequence(recurrent_input, all_num.cpu(), enforce_sorted=False, batch_first=True)
        out, hn = self.gru(packed_all_input)
        # hn = hn.squeeze()
        hn = hn[-1, :, :]
        out, input_size = pad_packed_sequence(out, batch_first=True)  # [B,L,F]

        # regressor
        result = self.regressor(wide_output, deep_output, hn)
        if re_target is not None:
            return result.squeeze(1), re_target
        if mid_num is not None:
            # Gather specific data from `out` tensor using indices from `mid_num`
            mid_num_temp = mid_num.view(-1, 1, 1)  # Reshape to match dimensions for gathering
            out_temp = out.gather(1, mid_num_temp.expand(-1, 1, out.shape[2])).reshape(batch_size, -1)
            mid_result = self.regressor(wide_output, deep_output, out_temp).squeeze(1)
            return result.squeeze(1), mid_result.squeeze(1)
        return result.squeeze(1)

    def flatten_parameters(self):
        """
        Calls flatten_parameters on all recurrent modules if applicable.
        """
        if isinstance(self.gru, nn.GRU):
            self.gru.flatten_parameters()


