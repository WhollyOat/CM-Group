import numpy as np
import torch
import itertools

def pairwise_distance(query_features, gallery_features):
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist

def ranking(query_feat, gallery_feat, query_ids, original_query_names, gallery_ids, original_gallery_names):
    dist = pairwise_distance(query_feat, gallery_feat)
    numpy_dist = dist.numpy()

    q_original_ids = np.unique(original_query_names)
    g_original_ids = np.unique(original_gallery_names)

    query_original_num = len(q_original_ids)
    gallery_original_num = len(g_original_ids)

    W = np.zeros((query_original_num, gallery_original_num)).astype(np.float16)
    
    for i in range(query_original_num):
        q_original_id = q_original_ids[i]
        id_index_q = np.isin(original_query_names, q_original_id)
        q_id_dist = numpy_dist[id_index_q, :]
        q_id = query_ids[id_index_q]
        q_id_num = len(q_id)

        for j in range(gallery_original_num):
            g_original_id = g_original_ids[j]
            id_index_g = np.isin(original_gallery_names, g_original_id)
            g_id = gallery_ids[id_index_g]
            g_id_num = len(g_id)
            qg_dist = q_id_dist[:,id_index_g]

            dist_container=[]

            if q_id_num > g_id_num:
                for q_permutation in itertools.permutations(range(q_id_num), g_id_num):
                    dist_perm = [qg_dist[q_permutation[u]][u] for u in range(g_id_num)]
                    dist_summ = sum(dist_perm) + (q_id_num - g_id_num)*max(dist_perm)
                    dist_container.append(dist_summ)
            else:
                for g_permutation in itertools.permutations(range(g_id_num), q_id_num):
                    dist_perm = [qg_dist[v][g_permutation[v]] for v in range(q_id_num)]
                    dist_summ = sum(dist_perm) + (g_id_num - q_id_num)*max(dist_perm)
                    dist_container.append(dist_summ)

            W[i,j]= min(dist_container)

    final_dist = W

    del dist
    del numpy_dist
    del W

    return final_dist