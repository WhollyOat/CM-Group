import os
import logging
import torch
import numpy as np
from sklearn.preprocessing import normalize
from .distance import ranking
from torch.nn import functional as F


def get_gallery_names(perm, cams, gp_ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1]
        for id in gp_ids:
            i = int(id.split('_')[0])
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(['camera{}/{}/{}'.format(cam, id, ins) for ins in instance_id])
    names = sorted(names)
    return names


def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]


def get_cmc(sorted_indices, query_ids, gallery_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):

        result_i = result[probe_index, :]

        # remove duplicated id in "stable" manner
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc


def get_mAP(sorted_indices, query_ids, gallery_ids):
    result = gallery_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
    
        result_i = result[probe_index, :]

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP


def eval_cmgroup(query_feats, query_ids, query_img_paths, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, perm, num_shots=1, num_trials=10):
    
    gallery_cams = [1, 2, 3]

    query_feats = F.normalize(query_feats, dim=1)
    original_query_names = np.array(['{}/{}/{}'.format(path.split('/')[-3], path.split('/')[-2].split('_')[0], path.split('/')[-1].split('.')[0]) for path in query_img_paths])
    q_original_gids = np.array([int(name.split('/')[1]) for name in np.unique(original_query_names)])

    gallery_indices = np.isin(gallery_cam_ids, gallery_cams)
    
    gallery_feats = gallery_feats[gallery_indices]
    gallery_feats = F.normalize(gallery_feats, dim=1)

    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_ids = gallery_ids[gallery_indices]

    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])
    original_gallery_names = np.array(['{}/{}/{}'.format(path.split('/')[-3], path.split('/')[-2].split('_')[0], path.split('/')[-1].split('.')[0]) for path in gallery_img_paths])
    
    gallery_gp_ids = [path.split('/')[-2] for path in gallery_img_paths]
    gallery_gp_id_set = np.unique(gallery_gp_ids)

    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names(perm, gallery_cams, gallery_gp_id_set, t, num_shots)
        flag = np.isin(gallery_names, names)

        g_feat = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_original_names = original_gallery_names[flag]
        g_original_gids = np.array([int(name.split('/')[1]) for name in np.unique(g_original_names)])

        dist_mat = ranking(query_feats, g_feat, query_ids, original_query_names,  g_ids, g_original_names)

        sorted_indices = np.argsort(dist_mat, axis=1)

        mAP += get_mAP(sorted_indices, q_original_gids, g_original_gids)
        cmc = get_cmc(sorted_indices, q_original_gids, g_original_gids)

        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]

    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100

    perf = 'num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(num_shots, r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20