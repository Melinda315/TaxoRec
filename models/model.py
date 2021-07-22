import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import load_npz
import scipy.sparse as sp
from utils.helper import default_device
import math
import torch.nn.functional as F
from geoopt import Lorentz
import geoopt.manifolds.lorentz.math as lorentz_math
from geoopt import PoincareBall
from geoopt import ManifoldParameter

import models.encoders as encoders

import time

eps = 1e-15


class TaxoRec(nn.Module):

    def __init__(self, users_items, args):
        super(TaxoRec, self).__init__()

        self.c = torch.tensor([args.c]).to(default_device())

        self.manifold = Lorentz(args.c)
        self.ball = PoincareBall(args.c)
        self.encoder = getattr(encoders, "HG")(args.c, args)

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.args = args

        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items,
                                      embedding_dim=args.embedding_dim).to(default_device())

        self.embedding.state_dict()['weight'].normal_(mean=0, std=args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight']), True)
        self.embedding.weight = ManifoldParameter(self.embedding.weight, self.manifold, True)

        tag_matrix = load_npz('data/' + args.dataset + '/item_tag_matrix.npz')
        tag_labels = tag_matrix.A
        tmp = np.sum(tag_labels, axis=1, keepdims=True)
        tag_labels = tag_labels / tmp
        self.num_tags = tag_labels.shape[1]

        self.T = nn.Embedding(num_embeddings=self.num_tags,
                              embedding_dim=args.dim).to(default_device())
        self.T.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.T.weight = nn.Parameter(self.manifold.expmap0(self.T.state_dict()['weight']), requires_grad=True)
        self.T.weight = ManifoldParameter(self.T.weight, self.manifold, requires_grad=True)

        self.ugr = nn.Embedding(num_embeddings=self.num_users,
                                embedding_dim=args.dim).to(default_device())
        self.ugr.state_dict()['weight'].normal_(mean=0, std=args.scale)
        self.ugr.weight = nn.Parameter(self.manifold.expmap0(self.ugr.state_dict()['weight']))
        self.ugr.weight = ManifoldParameter(self.ugr.weight, self.manifold, requires_grad=True)

        self.sps = torch.from_numpy(tag_labels).float().to(default_device())

        self.lam = args.lam

    def encode(self, adj):
        adj = adj.to(default_device())

        x1 = self.manifold.projx(self.embedding.weight)
        h_in = self.encoder.encode(x1, adj)

        emb_tag = self.manifold.projx(self.T.weight)
        emb_tag_in = self.manifold.projx(self.ugr.weight)
        emb_tag_weight = self.sps
        emb_tag_out = self.hyper_agg(emb_tag_weight, emb_tag)
        x2 = torch.cat([emb_tag_in, emb_tag_out], dim=0)
        h_gr = self.encoder.encode(x2, adj)

        h = torch.cat([h_in, h_gr], dim=-1)
        return h

    def decode(self, h_all, idx):
        h = h_all[:, :self.args.embedding_dim]
        emb_in = h[idx[:, 0]]
        emb_out = h[idx[:, 1]]
        sqdist = self.manifold.dist2(emb_in, emb_out, keepdim=True).clamp_max(50.0)
        assert not torch.isnan(sqdist).any()

        assert not torch.isinf(self.T.weight).any()
        assert not torch.isnan(self.T.weight).any()
        # print(self.T.weight)
        h2 = h_all[:, self.args.embedding_dim:]
        emb_tag_in = h2[idx[:, 0]]
        emb_tag_out = h2[idx[:, 1]]
        assert not torch.isnan(emb_tag_out).any()
        assert not torch.isinf(emb_tag_out).any()
        sqdist += self.manifold.dist2(emb_tag_in, emb_tag_out, keepdim=True).clamp_max(15.0)
        return sqdist

    def lorentz_factor(self, x, dim=-1, keepdim=False):
        """
        Parameters
        ----------
        x : tensor
            point on Klein disk
        c : float
            negative curvature
        dim : int
            dimension to calculate Lorenz factor
        keepdim : bool
            retain the last dim? (default: false)
        Returns
        -------
        tensor
            Lorenz factor
        """
        return 1 / torch.sqrt((1 - self.c.to(x.device) * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(eps))

    def hyper_agg(self, weights, x):
        x_p = self.l2p(x)
        x_k = self.p2k(x_p)
        gamma = self.lorentz_factor(x_k.detach(), dim=-1, keepdim=True)
        #
        mean = weights.matmul(gamma * x_k) / weights.matmul(gamma).clamp_min(eps)
        # return self.p2l(mean)

        return self.p2l(self.k2p(mean))

    def hyper_agg_uniform(self, x):
        x_p = self.l2p(x)
        x_k = self.p2k(x_p)
        gamma = self.lorentz_factor(x_k.detach(), dim=-1, keepdim=True)

        mean = (gamma * x_k) / (gamma.clamp_min(eps))

        return self.p2l(self.k2p(mean))

    def p2l(self, x):
        return lorentz_math.poincare_to_lorentz(x, k=self.c)

    def l2p(self, x):
        return lorentz_math.lorentz_to_poincare(x, k=self.c)

    def p2k(self, x):
        denom = 1 + self.c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    def k2p(self, x):
        denom = 1 + torch.sqrt(1 - self.c * x.pow(2).sum(-1, keepdim=True).clamp_min(eps))
        return x / denom

    def cluster_loss(self, tree, child_num=5):
        loss = 0
        for k in tree.keys():
            # each height
            if k == 0:
                continue
            node_list = tree[k]
            for i in range(len(node_list)):
                node = node_list[i].term_ids
                if len(node) == 0 or len(node) == 1:
                    continue
                try:
                    scores = node_list[i].scores.cuda(default_device())
                except Exception as e:
                    print(node_list[i].term_ids, k, i)
                scores = scores / scores.max()
                node_terms = self.T(torch.LongTensor(node).cuda(default_device()))

                center = self.hyper_agg(scores, node_terms).repeat(len(node)).view(len(node), -1)

                assert not torch.isnan(center).any()
                loss += ((node_terms - center) ** 2).sum()
        return loss

    def compute_loss(self, embeddings, child_num, triples, tree):
        assert not torch.isnan(triples).any()
        triples = triples.to(default_device())
        train_edges = triples[:, [0, 1]]

        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        pos_scores = self.decode(embeddings, train_edges)

        neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in
                           sampled_false_edges_list]
        neg_scores = torch.cat(neg_scores_list, dim=1)
        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)

        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

        if tree and self.lam > 0:
            cluster_loss = self.lam * self.cluster_loss(tree, child_num)
            loss += cluster_loss
        return loss

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :self.args.embedding_dim]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :self.args.embedding_dim]
            sqdist = self.manifold.dist2(emb_in, emb_out)

            emb_tag_in = h[:, self.args.embedding_dim:][i].repeat(num_items).view(num_items, -1)
            emb_tag_out = h[np.arange(num_users, num_users + num_items), self.args.embedding_dim:]
            sqdist += self.manifold.dist2(emb_tag_in, emb_tag_out)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix
