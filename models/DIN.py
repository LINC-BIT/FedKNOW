import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        if self.dim == 3:
            self.alpha = torch.zeros((num_features, 1)).cuda()
        elif self.dim == 2:
            self.alpha = torch.zeros((num_features,)).cuda()

    def forward(self, x):
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)

        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x

        return out


class MLP(nn.Module):

    def __init__(self, input_dimension, hidden_size, target_dimension=1, activation_layer='dice'):
        super().__init__()

        dimension_pair = [input_dimension] + hidden_size
        layers = []
        for i in range(len(dimension_pair) - 1):
            layers.append(nn.Linear(dimension_pair[i], dimension_pair[i + 1]))

            if activation_layer == 'dice':
                layers.append(Dice(dimension_pair[i + 1]))
            else:
                layers.append(nn.BatchNorm1d(dimension_pair[i + 1]))
                layers.append(nn.PReLU())

        layers.append(nn.Linear(hidden_size[-1], target_dimension))
        layers.insert(0, nn.BatchNorm1d(input_dimension))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class InputEmbedding(nn.Module):

    def __init__(self, n_uid, n_mid, n_cid, embedding_dim):
        super().__init__()
        self.user_embedding_unit = nn.Embedding(n_uid, embedding_dim)
        self.material_embedding_unit = nn.Embedding(n_mid, embedding_dim)
        self.category_embedding_unit = nn.Embedding(n_cid, embedding_dim)

    def forward(self, user, material, category, material_historical, category_historical,
                material_historical_neg, category_historical_neg, neg_smaple=False):
        user_embedding = self.user_embedding_unit(user)

        material_embedding = self.material_embedding_unit(material)
        material_historical_embedding = self.material_embedding_unit(material_historical)

        category_embedding = self.category_embedding_unit(category)
        category_historical_embedding = self.category_embedding_unit(category_historical)

        material_historical_neg_embedding = self.material_embedding_unit(
            material_historical_neg) if neg_smaple else None
        category_historical_neg_embedding = self.category_embedding_unit(
            category_historical_neg) if neg_smaple else None

        ans = [user_embedding, material_historical_embedding, category_historical_embedding,
               material_embedding, category_embedding, material_historical_neg_embedding,
               category_historical_neg_embedding]
        return tuple(map(lambda x: x.squeeze() if x != None else None, ans))


class AttentionLayer(nn.Module):

    def __init__(self, embedding_dim, hidden_size, activation_layer='dice'):
        super().__init__()

        dimension_pair = [embedding_dim * 8] + hidden_size
        layers = []
        for i in range(len(dimension_pair) - 1):
            layers.append(nn.Linear(dimension_pair[i], dimension_pair[i + 1]))

            if activation_layer == 'dice':
                layers.append(Dice(dimension_pair[i + 1], 3))
            else:
                layers.append(nn.BatchNorm1d(dimension_pair[i + 1]))
                layers.append(nn.PReLU())

        layers.append(nn.Linear(hidden_size[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, query, fact, mask, return_scores=False):
        B, T, D = fact.shape

        query = torch.ones((B, T, 1)).type(query.type()) * query.view((B, 1, D))
        # query = query.view(-1).expand( T, -1).view( T, B, D).permute( 1, 0, 2)

        combination = torch.cat([fact, query, fact * query, query - fact], dim=2)

        scores = self.model(combination).squeeze()
        scores = torch.where(mask == 1, scores, torch.ones_like(scores) * (-2 ** 31))

        scores = (scores.softmax(dim=-1) * mask).view((B, 1, T))

        if return_scores: return scores.squeeze()
        return torch.matmul(scores, fact).squeeze()


class DIN(nn.Module):
    def __init__(self, n_uid, n_mid, n_cid, embedding_dim, ):
        super().__init__()

        self.embedding_layer = InputEmbedding(n_uid, n_mid, n_cid, embedding_dim)
        self.attention_layer = AttentionLayer(embedding_dim, hidden_size=[80, 40], activation_layer='dice')
        self.output_layer = MLP(embedding_dim * 7, [200, 80], 1, 'dice')

    def forward(self, data, neg_sample=False):
        user, material_historical, category_historical, mask, sequential_length, material, category, \
        material_historical_neg, category_historical_neg = data

        user_embedding, material_historical_embedding, category_historical_embedding, \
        material_embedding, category_embedding, material_historical_neg_embedding, category_historical_neg_embedding = \
            self.embedding_layer(user, material, category, material_historical, category_historical,
                                 material_historical_neg, category_historical_neg, neg_sample)

        item_embedding = torch.cat([material_embedding, category_embedding], dim=1)
        item_historical_embedding = torch.cat([material_historical_embedding, category_historical_embedding], dim=2)

        item_historical_embedding_sum = torch.matmul(mask.unsqueeze(dim=1),
                                                     item_historical_embedding).squeeze() / sequential_length.type(
            mask.type()).unsqueeze(dim=1)

        attention_feature = self.attention_layer(item_embedding, item_historical_embedding, mask)

        # combination = torch.cat( [ user_embedding, item_embedding, item_historical_embedding_sum, attention_feature ], dim = 1)
        combination = torch.cat([user_embedding, item_embedding, item_historical_embedding_sum,
                                 # item_embedding * item_historical_embedding_sum,
                                 attention_feature], dim=1)

        scores = self.output_layer(combination)

        return scores.squeeze()