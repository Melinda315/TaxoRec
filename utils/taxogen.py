import torch
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from scipy.sparse import load_npz


def cos_sim(a, b):
    # sim = torch.cosine_similarity(a, b)
    sim = torch.sum((a - b) ** 2, dim=1)
    return sim


class TaxGenNode(object):

    def __init__(self, term_ids, user_ids, item_ids, parent, item_tags):
        self.term_ids = term_ids
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.scores = None
        self.parent = parent
        self.siblings = []

        if len(term_ids) > 0:
            self.tfd = item_tags[item_ids].sum()
            self.tf = item_tags[item_ids].sum(dim=0)  # 1 * term_num
            self.pop_scores = self.calc_pop()[term_ids]
            self.scores = torch.sqrt(self.pop_scores)
            self.df = item_tags[item_ids].sum(dim=0)
            self.max_df = self.df.max()
            self.dl = item_tags[item_ids].sum()
            self.avgdl = 0


            self.rel = self.bm25_df_paper()

            self.con = None
        else:
            self.rel = torch.zeros(item_tags.shape[1])
            self.dl = 0

    def bm25_df_paper(self, k=1.2, b=0.5, multiplier=3):
        df = self.df
        max_df = self.max_df
        tf = self.tf
        dl = self.dl
        avg_dl = self.avgdl
        score = tf * (k + 1) / (tf + k * (1 - b + b * (dl / avg_dl)))
        df_factor = torch.log2(1 + df) / torch.log2(1 + max_df)
        score *= df_factor
        score *= multiplier
        return score

    def calc_pop(self):
        return torch.log(self.tf + 1) / torch.log(self.tfd)

    def calc_scores(self):

        self.con = torch.exp(self.rel)

        divisor = 1
        for node in self.siblings:
            divisor += torch.exp(node.rel)


        self.con = self.con / divisor

        self.scores = self.scores * torch.sqrt(self.con[self.term_ids])

        return

def build_tree_gen(embeddings, child_num, tree_height, tag_labels):
    tree = dict()
    embeddings = embeddings.detach().cpu()
    tag_embeddings = embeddings
    tag_labels = (tag_labels > 0).float().cpu()
    # user_tags = rating_matrix.matmul(tag_labels)
    item_tags = tag_labels
    term_ids = torch.arange(tag_embeddings.shape[0])[tag_labels.sum(dim=0) > 0].tolist()
    item_ids = torch.arange(tag_labels.shape[0])[tag_labels.sum(dim=1) > 0]

    tree[0] = [TaxGenNode(term_ids, [], item_ids.tolist(), None, tag_labels)]
    for i in range(tree_height):
        tag_embeddings = embeddings
        nodes = tree[i]
        tree[i + 1] = []
        for j in range(len(nodes)):
            term_ids = nodes[j].term_ids

            tmp_term_ids = np.array(term_ids)
            if len(term_ids) < child_num:
                for child in range(child_num):
                    tree[i + 1].append(TaxGenNode([], [], [], nodes[j], tag_labels))
                continue

            while True:
                tmp_child = []
                X = tag_embeddings[tmp_term_ids].detach().cpu().numpy()
                clus = KMeans(n_clusters=child_num)
                clus.fit(X)
                labels = clus.labels_
                for child in range(child_num):
                    child_term_ids = tmp_term_ids[labels == child].tolist()
                    child_item_ids = torch.arange(item_tags.shape[0])[
                        item_tags[:, child_term_ids].sum(dim=1) > 0].tolist()
                    tmp_child.append(TaxGenNode(child_term_ids, [], child_item_ids, nodes[j], item_tags))
                total_dl = 0
                for child in range(child_num):
                    total_dl = tmp_child[child].dl

                avgdl = total_dl / child_num
                current_term_ids = []
                for child in range(child_num):
                    siblings = tree[i + 1][j * child_num:child] + tree[i + 1][child + 1:(j + 1) * child_num]
                    tmp_child[child].siblings = siblings
                    tmp_child[child].avgdl = avgdl
                    tmp_child[child].calc_scores()

                    for idx, term in enumerate(tmp_child[child].term_ids):
                        if tmp_child[child].scores[idx] >= 0.5: # modify
                            current_term_ids.append(term)

                if len(current_term_ids) == len(tmp_term_ids):
                    break
                tmp_term_ids = np.array(current_term_ids)
                if len(current_term_ids) < child_num:
                    tmp_child = []
                    for child in range(child_num):
                        tmp_child.append(TaxGenNode([], [], [], nodes[j], tag_labels))
                    break

            tree[i+1].extend(tmp_child)

    return tree
