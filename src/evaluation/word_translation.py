# codes in this script are modified from MUSE (https://github.com/facebookresearch/MUSE)

import os
import io
import sys
from logging import getLogger
import numpy as np
import torch

# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


logger = getLogger()


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()


def load_dictionary(params, file_name, word2id1, word2id2, eval=False):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency. 
    Uniqueness handled here.
    """
    path = os.path.join(params.dico_train_path, file_name) if eval==False else os.path.join(params.dico_eval_path, file_name)
    assert os.path.isfile(path)
    logger.info("loading dictionary from: {}".format(path))


    pairs = []
    src_w = []
    tgt_w = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                if params.sup_dict_size > 0 and eval==False:  #load only unique pairs
                    if word1 not in src_w and word2 not in tgt_w:
                        src_w.append(word1)
                        tgt_w.append(word2)
                        pairs.append((word1, word2))
                else:                         #load all pairs
                    pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    dico = dico[:params.sup_dict_size] if eval==False else dico
    return dico



def get_word_translation_accuracy(params, emb1, emb2, src2tgt=True):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    src_lang = params.src_dico.lang if src2tgt else params.tgt_dico.lang
    tgt_lang = params.tgt_dico.lang if src2tgt else params.src_dico.lang
    word2id1 =  params.src_dico.word2id if src2tgt else params.tgt_dico.word2id
    word2id2 =  params.tgt_dico.word2id if src2tgt else params.src_dico.word2id
    file_name = '%s-%s.5000-6500.txt' % (src_lang, tgt_lang)
    
    
    dico = load_dictionary(params, file_name, word2id1, word2id2, eval=True) # Return a torch tensor of size (n, 2) containing word2id indexes
    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    knn = params.csls_knn
    average_dist1 = get_nn_avg_dist(emb2, emb1, knn) 
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
    average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
    average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
    # queries / scores
    query = emb1[dico[:, 0]]
    scores = query.mm(emb2.transpose(0, 1))
    scores.mul_(2)
    scores.sub_(average_dist1[dico[:, 0]][:, None])
    scores.sub_(average_dist2[None, :])

    results = []
    top_matches = scores.topk(10, 1, True)[1]
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(torch.Tensor(list(matching.values())).numpy())
        logger.info("%i source words - CSLS-KNN-%i - Precision at k = %i: %f" %
                    (len(matching), knn, k, precision_at_k))
        if k==1:
            precision_at_1 = precision_at_k
    return precision_at_1

