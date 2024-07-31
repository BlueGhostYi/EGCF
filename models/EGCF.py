"""
Created on June 10, 2023,
PyTorch Implementation of EGCF
"""

import torch
from torch import nn
import utility.losses
import utility.tools
import utility.trainer


class EGCF(nn.Module):
    def __init__(self, config, dataset, device):
        super(EGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.temperature = float(self.config['temperature'])
        self.aggregate_mode = self.config['mode']

        self.user_embedding = None

        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.user_Graph = self.dataset.sparse_adjacency_matrix_R()  # sparse matrix
        self.user_Graph = utility.tools.convert_sp_mat_to_sp_tensor(self.user_Graph)  # sparse tensor
        self.user_Graph = self.user_Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy

        if self.aggregate_mode == 'parallel':
            self.Graph = self.dataset.sparse_adjacency_matrix()  # sparse matrix
            self.Graph = utility.tools.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
            self.Graph = self.Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy

        self.activation_layer = nn.Tanh()
        self.activation = nn.Sigmoid()

    def alternating_aggregate(self):
        item_embedding = self.item_embedding.weight

        all_user_embeddings = []
        all_item_embeddings = []

        for layer in range(int(self.config['GCN_layer'])):
            user_embedding = self.activation_layer(torch.sparse.mm(self.user_Graph, item_embedding))
            item_embedding = self.activation_layer(torch.sparse.mm(self.user_Graph.transpose(0, 1), user_embedding))

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)

        final_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        final_user_embeddings = torch.sum(final_user_embeddings, dim=1)

        final_item_embeddings = torch.stack(all_item_embeddings, dim=1)
        final_item_embeddings = torch.sum(final_item_embeddings, dim=1)

        return final_user_embeddings, final_item_embeddings

    def parallel_aggregate(self):
        item_embedding = self.item_embedding.weight
        user_embedding = self.activation_layer(torch.sparse.mm(self.user_Graph, item_embedding))

        all_embedding = torch.cat([user_embedding, item_embedding])

        all_embeddings = []

        for layer in range(int(self.config['GCN_layer'])):
            all_embedding = self.activation_layer(torch.sparse.mm(self.Graph, all_embedding))
            all_embeddings.append(all_embedding)

        final_all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.sum(final_all_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative):
        if self.aggregate_mode == 'parallel':
            all_user_embeddings, all_item_embeddings = self.parallel_aggregate()
        else:
            all_user_embeddings, all_item_embeddings = self.alternating_aggregate()

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = utility.losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = utility.losses.get_reg_loss(ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        ssl_user_loss = utility.losses.get_InfoNCE_loss(user_embedding, user_embedding, self.temperature)

        ssl_pos_loss = utility.losses.get_InfoNCE_loss(pos_embedding, pos_embedding,  self.temperature)

        ssl_inter_loss = utility.losses.get_InfoNCE_loss(user_embedding, pos_embedding, self.temperature)

        ssl_loss = self.ssl_lambda * (ssl_user_loss + ssl_pos_loss + ssl_inter_loss)

        loss_list = [bpr_loss, reg_loss, ssl_loss]

        return loss_list

    def get_embedding(self):
        if self.aggregate_mode == 'parallel':
            all_user_embeddings, all_item_embeddings = self.parallel_aggregate()
        else:
            all_user_embeddings, all_item_embeddings = self.alternating_aggregate()

        return all_user_embeddings, all_item_embeddings

    def get_rating_for_test(self, user):
        if self.aggregate_mode == 'parallel':
            all_user_embeddings, all_item_embeddings = self.parallel_aggregate()
        else:
            all_user_embeddings, all_item_embeddings = self.alternating_aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device):
        self.model = EGCF(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset

    def train(self):
        utility.trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device)
