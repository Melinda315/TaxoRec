import time
import traceback
from datetime import datetime

import numpy as np

from config import parser
from eval_metrics import recall_at_k
from models.model import TaxoRec
from geoopt.optim import RiemannianSGD
from torch import optim
from utils.data_generator import Data
from utils.helper import default_device, set_seed
from utils.sampler import WarpSampler
from utils.taxogen import build_tree_gen
import itertools, heapq
import torch


def train(model):

    save_path = 'data/' + args.dataset + '/' + args.model
    save_path += '.pt'
    print(save_path)
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Total number of parameters: {tot_params}")

    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    best_metric = [0, 0, 0, 0]

    pretrain_tag = 10

    # === Train model
    unchange = 0
    tree = None
    for epoch in range(args.epochs):
        if (epoch + 1) % pretrain_tag == 0 :
            item_attr_embeddings = model.l2p(model.T.weight.data)
            tree = build_tree_gen(item_attr_embeddings, args.child_num, 4, model.sps)
        avg_loss = 0.
        # === batch training
        t = time.time()
        for batch in range(num_batches):
            triples = sampler.next_batch()
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(data.adj_train_norm)
            train_loss = model.compute_loss(embeddings, args.child_num, triples, tree)
            loss = train_loss
            loss.backward()
            optimizer.step()
            avg_loss += train_loss / num_batches
        # === evaluate at the end of each batch
        avg_loss = avg_loss.detach().cpu().numpy()

        print(" ".join(['Epoch: {:04d}'.format(epoch),
                        '{:.3f}'.format(avg_loss),
                        'time: {:.4f}s'.format(time.time() - t)]), end=' ')
        print("")
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            start = time.time()
            embeddings = model.encode(data.adj_train_norm)
            pred_matrix = model.predict(embeddings, data)
            results = eval_rec(pred_matrix, data)
            recalls = results[0]
            if recalls[1] > best_metric[0]:
                best_metric[0] = recalls[1]
                best_metric[2] = results[1][1]
                torch.save(model.state_dict(), save_path)
                unchange = 0
            if recalls[2] > best_metric[1]:
                best_metric[1] = recalls[2]
                best_metric[3] = results[1][2]
                torch.save(model.state_dict(), save_path)
                unchange = 0
            else:
                unchange += args.eval_freq

            print('\t'.join([str(round(x, 4)) for x in results[0]]))
            print('\t'.join([str(round(x, 4)) for x in results[-1]]))
            print('best_eval', best_metric)
            if unchange == 200:
                print('best_eval', best_metric)
                break

    sampler.close()


def argmax_top_k(a, top_k=50):
    topk_score_items = []
    for i in range(len(a)):
        topk_score_item = heapq.nlargest(top_k, zip(a[i], itertools.count()))
        topk_score_items.append([x[1] for x in topk_score_item])
    return topk_score_items


def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len - 1]

        dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)


def eval_rec(pred_matrix, data):
    topk = 50
    pred_matrix[data.user_item_csr.nonzero()] = np.NINF
    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

    recall = []
    for k in [5, 10, 20, 50]:
        recall.append(recall_at_k(data.test_dict, pred_list, k))

    all_ndcg = ndcg_func([*data.test_dict.values()], pred_list)
    ndcg = [all_ndcg[x - 1] for x in [5, 10, 20, 50]]

    return recall, ndcg


if __name__ == '__main__':
    args = parser.parse_args()
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    print(args.dim, args.lr, args.weight_decay, args.margin, args.batch_size)
    print(args.scale, args.num_layers, args.network, args.model)
    print(default_device())

    # === fix seed
    set_seed(args.seed)

    # === prepare data
    data = Data(args.dataset, args.norm_adj, args.seed, args.test_ratio)
    total_edges = data.adj_train.count_nonzero()
    args.n_nodes = data.num_users + data.num_items
    args.feat_dim = args.embedding_dim

    # === negative sampler (iterator)
    sampler = WarpSampler((data.num_users, data.num_items), data.adj_train, args.batch_size, args.num_neg)

    model = TaxoRec((data.num_users, data.num_items), args)
    model = model.to(default_device())
    # tag_sampler = TagSampler(model.num_tags, args.batch_size)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('model is running on', next(model.parameters()).device)

    try:
        train(model)
    except Exception:
        sampler.close()
        traceback.print_exc()
