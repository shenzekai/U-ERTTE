import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# layer
class FeaturesLinear(nn.Module):

    def __init__(self, field_dims, output_dim):
        super(FeaturesLinear, self).__init__()
        self.fc = nn.Embedding(field_dims, output_dim)  # fc: Embedding:(610 + 193609, 1) 做一维特征的嵌入表示
        self.bias = nn.Parameter(torch.zeros((output_dim,)))  # Tensor: 1

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x:tensor([[554, 2320], [304, 3993]])
        # x: Tensor: 2048, 每个Tensor维度为2, x.new_tensor(self.offsets).unsqueeze(0): tensor([[0, 610]])
        # x: Tensor: [2048, 2]
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
        # 将所有嵌入的输出在最后一个维度上拼接起来
        return torch.cat(embedded_features, dim=1)


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = nn.Embedding(field_dims, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight.data)
        # embedding weight的初始化通过均匀分布的采用得到

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
        # 将所有嵌入的输出在最后一个维度上拼接起来
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


class LinkMultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):  # 512, 256
        super(LinkMultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))  # 512 256
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 3))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        # Reshape x to apply MLP independently to each field
        batch_size, num_fields, embed_dim = x.size()
        x = x.view(-1, embed_dim)  # Reshape to (batch_size * num_fields, embed_dim)
        x = self.mlp(x)  # Apply MLP
        x = x.view(batch_size, num_fields, -1)  # Restore shape to (batch_size, num_fields, output_dim)
        return x

# model
class MLPTTE(nn.Module):

    def __init__(self, FLAGS):
        super(MLPTTE, self).__init__()

        mlp_out_dim = 256
        lstm_hidden_size = 256

        # feature range
        feature_ranges = [FLAGS.num_components, FLAGS.highway_num, FLAGS.lane_num, FLAGS.reversed, FLAGS.oneway]
        embedding_dims = [16, 5, 4, 2, 2]
        all_embed_dim = 29

        self.feature_embeddings = LinkFeatureEmbedding(feature_ranges, embedding_dims)
        self.all_mlp = nn.Sequential(
            nn.Linear(all_embed_dim + 3, mlp_out_dim),
            nn.ReLU()
        )

        self.LinkReg = LinkMultiLayerPerceptron(lstm_hidden_size, (16,), dropout=0.15, output_layer=True)

    def forward(self, all_link_feature, all_flow, all_linkdistance, all_real, all_num, mid_num=None):
        all_feature_embedding = self.feature_embeddings(all_link_feature)
        all_input = torch.cat([all_feature_embedding, all_real, all_flow, all_linkdistance], dim=2)
        mlp_input = self.all_mlp(all_input)  # batch_size, seq_len, hidden_size
        mask = torch.arange(all_input.shape[1]).expand(all_input.shape[0], all_input.shape[1]).to(device) < all_num.unsqueeze(1)
        mlp_input = mlp_input * mask.unsqueeze(2).float()
        link_out = self.LinkReg(mlp_input)
        result = torch.sum(link_out, dim=1)
        if mid_num is not None:
            mid_mask = torch.arange(link_out.shape[1]).expand(link_out.shape[0], link_out.shape[1]).to(device) < mid_num.unsqueeze(1)
            mid_out = link_out * mid_mask.unsqueeze(2).float()
            mid_result = torch.sum(mid_out, dim=1)
            return result.squeeze(1), mid_result.squeeze(1)
        return result.squeeze(1)
