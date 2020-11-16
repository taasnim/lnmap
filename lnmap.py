import os
import time
import json
import random
import logging
import argparse
import torch
import numpy as np
import jsbeautifier
from tqdm import tqdm
from collections import OrderedDict

from params import load_args
from logger import create_logger
from src.trainer import Trainer
from src.models import build_model
from src.evaluation import Evaluator
from src.refinement import generate_new_dictionary_bidirectional
from src.evaluation.word_translation import get_word_translation_accuracy

opts = jsbeautifier.default_options()
logger = logging.getLogger(__name__)

def set_seed(params):
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)


def load_autoenc_weights(params, trainer, logger):
    path = os.path.join(params.autoenc_weights_path, params.src_lang + "-" + params.tgt_lang)
    if not os.path.exists(path):
        trainer.train_autoencoder_A(logger)
        trainer.train_autoencoder_B(logger)
        save_autoenc_weights(params, trainer, logger)
    else:    
        trainer.encoder_A.load_state_dict(torch.load(os.path.join(path, 'encoder_A.pth')))
        trainer.encoder_B.load_state_dict(torch.load(os.path.join(path, 'encoder_B.pth')))
        trainer.decoder_A.load_state_dict(torch.load(os.path.join(path, 'decoder_A.pth')))
        trainer.decoder_B.load_state_dict(torch.load(os.path.join(path, 'decoder_B.pth')))    
        logger.info("Loaded saved autoencoder weights from {}".format(path))

def save_autoenc_weights(params, trainer, logger):
    path = os.path.join(params.autoenc_weights_path, params.src_lang + "-" + params.tgt_lang)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(trainer.encoder_A.state_dict(), os.path.join(path, 'encoder_A.pth'))
    torch.save(trainer.encoder_B.state_dict(), os.path.join(path, 'encoder_B.pth'))
    torch.save(trainer.decoder_A.state_dict(), os.path.join(path, 'decoder_A.pth'))
    torch.save(trainer.decoder_B.state_dict(), os.path.join(path, 'decoder_B.pth'))
    logger.info("Saved autoencoder weights to {}".format(path))

def save_model_weights(params, trainer, src2tgt=True):
    path = params.model_weights_path + params.src_lang + "-" + params.tgt_lang
    if not os.path.exists(path):
        os.makedirs(path)
    post_fix = "_AB" if src2tgt else "_BA"
    torch.save(trainer.mapping_G.state_dict(), os.path.join(path, 'mapper_G'+post_fix+'.pth'))
    torch.save(trainer.mapping_F.state_dict(), os.path.join(path, 'mapper_F'+post_fix+'.pth'))
    torch.save(trainer.encoder_A.state_dict(), os.path.join(path, 'encoder_A'+post_fix+'.pth'))
    torch.save(trainer.encoder_B.state_dict(), os.path.join(path, 'encoder_B'+post_fix+'.pth'))
    torch.save(trainer.decoder_A.state_dict(), os.path.join(path, 'decoder_A'+post_fix+'.pth'))
    torch.save(trainer.decoder_B.state_dict(), os.path.join(path, 'decoder_B'+post_fix+'.pth'))


def main():
    params = load_args()
    
    logger = create_logger(os.path.join(params.exp_path, "lnmap-experiment.log"))
    logger.info("{}".format(jsbeautifier.beautify(json.dumps(params.__dict__), opts)))
    set_seed(params)

    src_emb, tgt_emb, mapping_G, mapping_F, encoder_A, decoder_A, encoder_B, decoder_B = build_model(params)
    trainer = Trainer(src_emb, tgt_emb, mapping_G, mapping_F, encoder_A, decoder_A, encoder_B, decoder_B, params)
    evaluator = Evaluator(trainer)

    trainer.load_training_dico(logger)
    trainer.load_training_dico(logger, src2tgt=False)
    logger.info("Source train dico shape: {}".format(trainer.dico_AB.shape))
    logger.info("Target train dico shape: {}".format(trainer.dico_BA.shape))
    trainer.dico_AB_original = trainer.dico_AB.clone()
    trainer.dico_BA_original = trainer.dico_BA.clone()

    if params.load_autoenc_weights:
        load_autoenc_weights(params, trainer, logger)
    else:
        trainer.train_autoencoder_A(logger)
        trainer.train_autoencoder_B(logger)
        if params.save_autoenc_weights:
            save_autoenc_weights(params, trainer, logger)
    
    # Source to Target Training
    logger.info("\n \n Training for {} to {}".format( params.src_lang, params.tgt_lang))
    for i in range(params.iteration):
        logger.info("\n\n***Iteration: {}".format(i))
        trainer.train_A2B()
        
        trainer.set_eval()
        precision_at_1  = get_word_translation_accuracy(params, 
                trainer.mapping_G(trainer.encoder_A(trainer.src_emb.weight.data).data).data,
                trainer.encoder_B(trainer.tgt_emb.weight.data).data,
                src2tgt=True
            )
   
        emb1 = (trainer.mapping_G(trainer.encoder_A(trainer.src_emb.weight.data)).data)[0:params.dico_max_rank]
        emb2 = (trainer.encoder_B(trainer.tgt_emb.weight.data).data)[0:params.dico_max_rank]
        emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

        all_pairs, all_scores  = generate_new_dictionary_bidirectional(emb1, emb2)
        
        add_size = params.induced_dico_c*(i+1)
        trainer.dico_AB = torch.cat((trainer.dico_AB_original, all_pairs[:add_size].cuda()), 0)
        logger.info("New train dictionary shape: {}".format(trainer.dico_AB.shape))

    if params.save_model_weights:
        save_model_weights(params, trainer, src2tgt=True)


    # Target to Source Training
    logger.info("\n \n Training for {} to {}".format(params.tgt_lang, params.src_lang))
    n_iter = 0
    for i in range(params.iteration):
        logger.info("\n\n***Iteration: {}".format(i))
        trainer.train_B2A()
        
        trainer.set_eval()
        precision_at_1 = get_word_translation_accuracy(params,
                trainer.mapping_F(trainer.encoder_B(trainer.tgt_emb.weight.data).data).data, 
                trainer.encoder_A(trainer.src_emb.weight.data).data,
                src2tgt=False
            )
   
        emb1 = ((trainer.encoder_A(trainer.src_emb.weight.data)).data)[0:params.dico_max_rank]
        emb2 = (trainer.mapping_F(trainer.encoder_B(trainer.tgt_emb.weight.data)).data)[0:params.dico_max_rank]
        emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

        all_pairs, all_scores = generate_new_dictionary_bidirectional(emb2, emb1)
        
        add_size = params.induced_dico_c*(i+1)
        trainer.dico_BA = torch.cat((trainer.dico_BA_original, all_pairs[:add_size].cuda()), 0)
        logger.info("New train dictionary shape: {}".format(trainer.dico_AB.shape))
 
    if params.save_model_weights:
        save_model_weights(params, trainer, src2tgt=False)
        

if __name__ == "__main__":
    main()