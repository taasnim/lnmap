import os
import random
import scipy
import scipy.linalg
import numpy as np
import collections
from tqdm import tqdm
from logging import getLogger

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

from .utils import get_optimizer
from .dico_builder import build_dictionary
from .evaluation.word_translation import load_dictionary

logger = getLogger()


def mapping_loss(real, fake):
    return F.mse_loss(real, fake)

def cycle_loss(real, cycled):
    return F.mse_loss(real, cycled)

def reconstruction_loss(real, recons):
    return F.mse_loss(real, recons) 

class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping_G, mapping_F, encoder_A, decoder_A, encoder_B, decoder_B, params):
        self.params = params
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.encoder_A = encoder_A
        self.decoder_A = decoder_A
        self.encoder_B = encoder_B
        self.decoder_B = decoder_B
        self.mapping_G = mapping_G
        self.mapping_F = mapping_F
        self.src_dico = params.src_dico
        self.tgt_dico = params.tgt_dico
        self.dataset = params.dataset

        # optimizers
        optim_fn, optim_params = get_optimizer(params.map_optimizer)
        self.map_optimizer_G = optim_fn(mapping_G.parameters(), **optim_params)
        self.map_optimizer_F = optim_fn(mapping_F.parameters(), **optim_params)
        
        optim_fn, optim_params = get_optimizer(params.autoenc_optimizer)
        self.encoder_A_optimizer = optim_fn(encoder_A.parameters(), **optim_params)
        self.decoder_A_optimizer = optim_fn(decoder_A.parameters(), **optim_params)
        self.encoder_B_optimizer = optim_fn(encoder_B.parameters(), **optim_params)
        self.decoder_B_optimizer = optim_fn(decoder_B.parameters(), **optim_params)

    def set_eval(self):        
        self.encoder_A.eval()
        self.decoder_A.eval()
        self.encoder_B.eval()
        self.decoder_B.eval()
        self.mapping_G.eval()
        self.mapping_F.eval()
    
    def set_train(self):
        self.encoder_A.train()
        self.decoder_A.train()
        self.encoder_B.train()
        self.decoder_B.train()
        self.mapping_G.train()
        self.mapping_F.train()       
    


    def train_autoencoder_A(self, logger):
        logger.info("**Training Source Autoencoder**")
        self.encoder_A.train()
        self.decoder_A.train()
        bs = self.params.batch_size
        for epoch in tqdm(range(self.params.autoenc_epochs)):
            total_loss = 0
            num_batches = 0
            for n_iter in range(0, self.params.autoenc_epoch_size, bs):                
                ids = torch.LongTensor(bs).random_(len(self.src_dico)) # select random word IDs
                if self.params.cuda:
                    ids = ids.cuda()
                emb = self.src_emb(ids) # get word embeddings
                preds = self.decoder_A(self.encoder_A(emb.data))
                
                loss = F.mse_loss(emb.data, preds)
                total_loss += loss.detach().item()
                num_batches += 1

                self.encoder_A_optimizer.zero_grad()
                self.decoder_A_optimizer.zero_grad()
                loss.backward()
                self.encoder_A_optimizer.step()
                self.decoder_A_optimizer.step()

    def train_autoencoder_B(self, logger):
        logger.info("**Training Target Autoencoder**")
        self.encoder_B.train()
        self.decoder_B.train()
        bs = self.params.batch_size
        for epoch in tqdm(range(self.params.autoenc_epochs)):
            total_loss = 0
            num_batches = 0
            for n_iter in range(0, self.params.autoenc_epoch_size, bs):              
                ids = torch.LongTensor(bs).random_(len(self.tgt_dico)) # select random word IDs
                if self.params.cuda:
                    ids = ids.cuda()                
                emb = self.tgt_emb(ids) # get word embeddings
                preds = self.decoder_B(self.encoder_B(emb.data))
                
                loss = F.mse_loss(emb.data, preds)
                total_loss += loss.detach().item()
                num_batches += 1

                self.encoder_B_optimizer.zero_grad()
                self.decoder_B_optimizer.zero_grad()
                loss.backward()
                self.encoder_B_optimizer.step()
                self.decoder_B_optimizer.step()

    def train_A2B(self):
        scheduler_G = StepLR(self.map_optimizer_G, step_size=30, gamma=0.1)
        scheduler_F = StepLR(self.map_optimizer_F, step_size=30, gamma=0.1)

        self.set_train()
        bs = self.params.batch_size
        for epoch in (range(self.params.mapper_epochs)):
            dico_AB = self.dico_AB
            shuffled_indices = torch.randperm(dico_AB.size(0))
            for count in range(0, dico_AB.size(0), bs):
                ids = shuffled_indices[count:count+bs]
                ids = ids.cuda() if self.params.cuda else ids
                
                # in embedding space
                real_x = self.src_emb.weight[dico_AB[ids][:, 0]]
                real_y = self.tgt_emb.weight[dico_AB[ids][:, 1]]
                # in code space
                real_zx = self.encoder_A(real_x)
                real_zy = self.encoder_B(real_y)
                # in other's code space
                fake_zy = self.mapping_G(real_zx)
                # back-translated to self code space
                cycled_zx = self.mapping_F(fake_zy)
                # reconstruction of embedding space
                x_hat = self.decoder_A(cycled_zx)                        

                total_mapping_loss = mapping_loss(real_zy, fake_zy)
                total_cycle_loss = cycle_loss(real_zx, cycled_zx) 
                total_recons_loss = reconstruction_loss(real_x, x_hat) 

                total_loss = self.params.mapping_lambda*total_mapping_loss + self.params.cycle_lambda*total_cycle_loss + self.params.reconstruction_lambda*total_recons_loss
                
                # optim
                self.encoder_A_optimizer.zero_grad()
                self.decoder_A_optimizer.zero_grad()
                self.map_optimizer_G.zero_grad()
                self.map_optimizer_F.zero_grad()
                total_loss.backward()
                self.map_optimizer_G.step()
                self.map_optimizer_F.step()
                self.encoder_A_optimizer.step()
                self.decoder_A_optimizer.step()
                
            scheduler_G.step()
            scheduler_F.step()
                
                
    def train_B2A(self):
        scheduler_G = StepLR(self.map_optimizer_G, step_size=30, gamma=0.1)
        scheduler_F = StepLR(self.map_optimizer_F, step_size=30, gamma=0.1)
        
        self.set_train()
        bs = self.params.batch_size
        for epoch in (range(self.params.mapper_epochs)):
            dico_BA = self.dico_BA
            shuffled_indices = torch.randperm(dico_BA.size(0))
            for count in range(0, dico_BA.size(0), bs):
                ids = shuffled_indices[count:count+bs]
                ids = ids.cuda() if self.params.cuda else ids
                    
                # in embedding space
                real_x = self.src_emb.weight[dico_BA[ids][:, 1]]
                real_y = self.tgt_emb.weight[dico_BA[ids][:, 0]]
                # in code space
                real_zx = self.encoder_A(real_x)
                real_zy = self.encoder_B(real_y)
                # in other's code space
                fake_zx = self.mapping_F(real_zy)
                # back-translated to self code space
                cycled_zy = self.mapping_G(fake_zx)
                # reconstruction of embedding space
                y_hat = self.decoder_B(cycled_zy)
                
                total_mapping_loss = mapping_loss(real_zx, fake_zx) 
                total_cycle_loss = cycle_loss(real_zy, cycled_zy) 
                total_recons_loss = reconstruction_loss(real_y, y_hat) 

                total_loss = self.params.mapping_lambda*total_mapping_loss + self.params.cycle_lambda*total_cycle_loss + self.params.reconstruction_lambda*total_recons_loss
                
                # optim
                self.encoder_B_optimizer.zero_grad()
                self.decoder_B_optimizer.zero_grad()
                self.map_optimizer_G.zero_grad()
                self.map_optimizer_F.zero_grad()
                total_loss.backward()
                self.map_optimizer_G.step()
                self.map_optimizer_F.step()
                self.encoder_B_optimizer.step()
                self.decoder_B_optimizer.step()
                
            scheduler_G.step()
            scheduler_F.step()
                
                

    def build_dictionary_AB(self):
        """
        Build a dictionary from aligned embeddings for A->B.
        """
        self.encoder_A.eval()
        self.decoder_A.eval()
        self.encoder_B.eval()
        self.decoder_B.eval()
        src_emb = self.mapping_G(self.encoder_A(self.src_emb.weight.data)).data
        tgt_emb = self.encoder_B(self.tgt_emb.weight.data).data

        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico_AB = build_dictionary(src_emb, tgt_emb, self.params)


    def build_dictionary_BA(self):
        """
        Build a dictionary from aligned embeddings for B->A.
        """
        self.encoder_A.eval()
        self.decoder_A.eval()
        self.encoder_B.eval()
        self.decoder_B.eval()
        src_emb = self.encoder_A(self.src_emb.weight.data).data
        tgt_emb = self.mapping_F(self.encoder_B(self.tgt_emb.weight.data)).data

        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico_BA = build_dictionary(tgt_emb, src_emb, self.params)

 
    def load_training_dico(self, logger, src2tgt=True):
        """
        Load training dictionary. Unique/all
        """
        word2id1 = self.src_dico.word2id if src2tgt else self.tgt_dico.word2id
        word2id2 = self.tgt_dico.word2id if src2tgt else self.src_dico.word2id
        src_lang = self.params.src_lang if src2tgt else self.params.tgt_lang
        tgt_lang = self.params.tgt_lang if src2tgt else self.params.src_lang
        if self.dataset == "muse":
            file_name = '%s-%s.0-5000.txt' % (src_lang, tgt_lang) 
        elif self.dataset == "vecmap":
            file_name = '%s-%s.train.txt' % (src_lang, tgt_lang) 
        else:
            raise NotImplementedError
        
        if src2tgt:
            self.dico_AB = load_dictionary(self.params, file_name, word2id1, word2id2)
            self.dico_AB = self.dico_AB.cuda() if self.params.cuda else self.dico_AB
        else:
            self.dico_BA = load_dictionary(self.params, file_name, word2id1, word2id2)
            self.dico_BA = self.dico_BA.cuda() if self.params.cuda else self.dico_BA
        
        logger.info("loading dictionary from: {}".format(os.path.join(self.params.dico_train_path, file_name)))
             

    def save_best(self, to_log, metric, AB=True):
        """
        Save the best model for the given validation metric
        """
        # best mapping for the given validation criterion
        if AB == True:
            if to_log[metric] > self.best_valid_metric_AB:
                print("###**Saving AB**##")
                self.best_valid_metric_AB = to_log[metric]
                torch.save(self.mapping_G.state_dict(), os.path.join(
                    self.params.exp_path, 'mapping_G_AB.pth'))
                torch.save(self.encoder_A.state_dict(), os.path.join(
                    self.params.exp_path, 'encoder_A_AB.pth'))
                torch.save(self.encoder_B.state_dict(), os.path.join(
                    self.params.exp_path, 'encoder_B_AB.pth'))
        else:
            if to_log[metric] > self.best_valid_metric_BA:
                print("###**Saving BA**##")
                self.best_valid_metric_BA = to_log[metric]
                torch.save(self.mapping_F.state_dict(), os.path.join(
                    self.params.exp_path, 'mapping_F_BA.pth'))
                torch.save(self.encoder_A.state_dict(), os.path.join(
                    self.params.exp_path, 'encoder_A_BA.pth'))
                torch.save(self.encoder_B.state_dict(), os.path.join(
                    self.params.exp_path, 'encoder_B_BA.pth'))

    def reload_best(self, path=None, AB=True):
        """
        Reload the best saved params
        """
        if AB == True:
            self.mapping_G.load_state_dict(torch.load(os.path.join(
                self.params.exp_path, 'mapping_G_AB.pth') if path == None else os.path.join(path, 'mapping_G_AB.pth')))
            self.encoder_A.load_state_dict(torch.load(os.path.join(
                self.params.exp_path, 'encoder_A_AB.pth') if path == None else os.path.join(path, 'encoder_A_AB.pth')))
            self.encoder_B.load_state_dict(torch.load(os.path.join(
                self.params.exp_path, 'encoder_B_AB.pth') if path == None else os.path.join(path, 'encoder_B_AB.pth')))
        else:
            self.mapping_F.load_state_dict(torch.load(os.path.join(
                self.params.exp_path, 'mapping_F_BA.pth') if path == None else os.path.join(path, 'mapping_F_BA.pth')))
            self.encoder_A.load_state_dict(torch.load(os.path.join(
                self.params.exp_path, 'encoder_A_BA.pth') if path == None else os.path.join(path, 'encoder_A_BA.pth')))
            self.encoder_B.load_state_dict(torch.load(os.path.join(
                self.params.exp_path, 'encoder_B_BA.pth') if path == None else os.path.join(path, 'encoder_B_BA.pth')))
           

                
                
