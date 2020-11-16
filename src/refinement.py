import torch
import numpy as np
from .utils import normalize_embeddings, get_nn_avg_dist
import torch.nn as nn


cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)


def generate_new_dictionary_bidirectional(emb1, emb2, dico_max_rank=15000):
    '''
    build a dictionary from aligned embeddings
    '''
    emb1 = emb1.cuda()
    emb2 = emb2.cuda()
    bs = 128
    all_scores_S2T = []
    all_targets_S2T = []
    all_scores_T2S = []
    all_targets_T2S = []
    # number of source words to consider
    n_src = dico_max_rank
    knn = 10
    
    average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn)) #emb1 is query here
    average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn)) #emb2 is query here
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)
    
    ## emb1 to emb2
    for i in range(0, n_src, bs):
        scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

        all_scores_S2T.append(best_scores.cpu())
        all_targets_S2T.append(best_targets.cpu())

    all_scores_S2T = torch.cat(all_scores_S2T, 0)
    all_targets_S2T = torch.cat(all_targets_S2T, 0)

    all_pairs_S2T = torch.cat([torch.arange(0, all_targets_S2T.size(0)).long().unsqueeze(1),
                                all_targets_S2T[:, 0].unsqueeze(1)], 1)
    all_pairs_S2T = set([(a, b) for a, b in all_pairs_S2T.numpy()]) # converting to set to find intersection
    
    # emb2 to emb1
    for i in range(0, n_src, bs):
        scores = emb1.mm(emb2[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist2[i:min(n_src, i + bs)][:, None] + average_dist1[None, :])
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

        all_scores_T2S.append(best_scores.cpu())
        all_targets_T2S.append(best_targets.cpu())

    all_scores_T2S = torch.cat(all_scores_T2S, 0)
    all_targets_T2S = torch.cat(all_targets_T2S, 0)

    all_pairs_T2S = torch.cat([all_targets_T2S[:, 0].unsqueeze(1),
        torch.arange(0, all_targets_T2S.size(0)).long().unsqueeze(1)], 1) #making dictionary similar to emb1 to emb2
    all_pairs_T2S = set([(a, b) for a, b in all_pairs_T2S.numpy()])

    final_pairs = torch.LongTensor(list(all_pairs_S2T & all_pairs_T2S)) #Take the common i.e. intersection of the 2 dictionary
    
    scores = cosine_similarity(emb1[final_pairs[:,0]],emb2[final_pairs[:,1]])
    scores, reordered = scores.sort(0, descending=True)
    final_pairs = final_pairs[reordered]
    
    return final_pairs, scores


def symmetric_reweighting(src_emb, tgt_emb, src_indices, trg_indices):
    '''
    Symmetric reweighting refinement procedure
    '''
    xw = (src_emb.weight.clone()).data
    zw = (tgt_emb.weight.clone()).data
    
    _ = normalize_embeddings(xw.data, 'renorm,center,renorm')
    _ = normalize_embeddings(zw.data, 'renorm,center,renorm')

    # STEP 1: Whitening
    def whitening_transformation(m):
        u, s, v = torch.svd(m) 
        return v.mm(torch.diag(1/(s))).mm(v.t())

    wx1 = whitening_transformation(xw[src_indices]).type_as(xw)
    wz1 = whitening_transformation(zw[trg_indices]).type_as(zw)

    xw = xw.mm(wx1)
    zw = zw.mm(wz1)

    # STEP 2: Orthogonal mapping
    wx2, s, wz2 = torch.svd(xw[src_indices].t().mm(zw[trg_indices]), some=False)
    
    xw = xw.mm(wx2)
    zw = zw.mm(wz2)

    # STEP 3: Re-weighting
    xw *= s**0.5
    zw *= s**0.5 

    # STEP 4: De-whitening
    xw = xw.mm(wx2.transpose(0, 1).mm(torch.inverse(wx1)).mm(wx2))
    zw = zw.mm(wz2.transpose(0, 1).mm(torch.inverse(wz1)).mm(wz2))
    
    return xw, zw


