# lnmap

This repository contains the code of our paper [LNMap: Departures from Isomorphic Assumption in Bilingual Lexicon Induction Through Non-Linear Mapping in Latent Space](https://www.aclweb.org/anthology/2020.emnlp-main.215/) (EMNLP 2020).

![lnmap](https://taasnim.github.io/img/lnmap/model.png)


## Requirements

- Python 3.6 or higher with NumPy/SciPy
- PyTorch 1.4 or higher
- tqdm
- jsbeautifier


## Get Datasets

### 1. Get Monolingual Word Embeddings

### 2. Get evaluation datasets

We evaluate our models on two popularly used datasets: [MUSE (Conneau et al., 2018)](https://github.com/facebookresearch/MUSE) and [VecMap (Dinu et al., 2015)](https://github.com/artetxem/vecmap/). Please visit the respective repository for downloading the datasets.


## How to run
To run En<->Bn model on a single GPU:

```
export CUDA_VISIBLE_DEVICES="0"

src_lang="en"
tgt_lang="bn"
dataset="muse"
sup_dict_size=1000
autoenc_dim=350
seed=1000

root_project_folder="./dumped" 
project_folder=$root_project_folder"/"$dataset

folder_address="sup-dict-"$sup_dict_size"/"$autoenc_dim"-"$autoenc_dim
folder_address=$folder_address"/"$src_lang"-"$tgt_lang
folder_address=$folder_address"--sup_dict-"$sup_dict_size
folder_address=$folder_address"--seed-"$seed
OUT_ADDRESS=$project_folder"/"$folder_address
            
mkdir -p $OUT_ADDRESS

python -u lnmap.py \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --src_emb "path/to/word-embeddings/folder/wiki."$src_lang".vec" \
    --tgt_emb "path/to/word-embeddings/folder/wiki."$tgt_lang".vec" \
    --sup_dict_size $sup_dict_size \
    --emb_dim_autoenc_A $autoenc_dim \
    --emb_dim_autoenc_B $autoenc_dim \
    --dico_train_path "path/to/train-dictionary/folder/" \
    --dico_eval_path "path/to/eval-dictionary/folder/" \
    --exp_path $OUT_ADDRESS\
    --seed $seed \
    --nonlinear_autoenc \
    --nonlinear_mapper \
    --dico_max_rank 50000 
            
```

You will get the word translation accuracies at 3 different precision (1, 5, 10) for EN-BN and Bn-EN.



## Citation
Please cite our paper if you found the resources in this repository useful.
```bibtex
@inproceedings{mohiuddin-etal-2020-lnmap,
    title = "{LNM}ap: Departures from Isomorphic Assumption in Bilingual Lexicon Induction Through Non-Linear Mapping in Latent Space",
    author = "Mohiuddin, Tasnim  and
      Bari, M Saiful  and
      Joty, Shafiq",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.215",
    pages = "2712--2723"
}
```
