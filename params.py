import os
import argparse


def data_params(parser):
    group = parser.add_argument_group('Dataset params.')
    group.add_argument("--src_lang", type=str, default='en', help="Source language")
    group.add_argument("--tgt_lang", type=str, default='es', help="Target language")
    group.add_argument("--src_emb", type=str, default="data/", help="Source embedding file path")
    group.add_argument("--tgt_emb", type=str, default="data/", help="Target embedding file path")
    group.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    group.add_argument("--max_vocab_A", type=int, default=200000, help="Maximum vocabulary size for source (-1 to disable)")
    group.add_argument("--max_vocab_B", type=int, default=200000, help="Maximum vocabulary size for target (-1 to disable)")
    group.add_argument("--dataset", type=str, default="muse", help="muse or vecmap?")
    group.add_argument("--dico_train_path", type=str, default="", help="Path to training dictionary folder. Give only original folder.")
    group.add_argument("--dico_eval_path", type=str, default="", help="Path to evaluation dictionary folder. Give only original folder.")
    

def autoenc_params(parser):
    group = parser.add_argument_group('Autoencoder params.')
    group.add_argument("--hidden_dim", type=int, default=400, help="hidden dimension")
    group.add_argument("--emb_dim_autoenc_A", type=int, default=350, help="Embedding dimension in bottle kneck of autoencoder")
    group.add_argument("--emb_dim_autoenc_B", type=int, default=350, help="Embedding dimension in bottle kneck of autoencoder")


def mapping_params(parser):
    group = parser.add_argument_group('Mapping params.')
    group.add_argument("--map_id_init", type=bool, default=True, help="Initialize the mapping as an identity matrix")
    group.add_argument("--mapper_hidden_dim", type=int, default=400, help="hidden dimension in nonlinear mapper")
    group.add_argument("--mapper_nonlinearity", type=int, default=0, help="Nonlinearity in mapper hidden layer: 0-tanh, 1-relu, 2-lrelu, 3-prelu")
    group.add_argument("--mapper_dropout", type=float, default=0.1, help="Nonlinear Mapper dropout")
    

def training_params(parser):
    group = parser.add_argument_group('Training params.')
    group.add_argument("--load_autoenc_weights", type=bool, default=False, help="Use saved autoencoder weights instead of training them")
    group.add_argument("--save_autoenc_weights", type=bool, default=False, help="Save trained autoencoder weights")
    group.add_argument("--autoenc_weights_path", type=str, default="", help="Where to store autoencoder weights")
    group.add_argument("--model_weights_path", type=str, default="", help="Where to store autoencoder weights")
    group.add_argument("--load_model_weights", type=bool, default=False, help="Use saved model weights instead of training them")
    group.add_argument("--save_model_weights", type=bool, default=False, help="Save trained model weights")
    group.add_argument("--sup_dict_size", type=int, default=1000, help="Number of initial seeds from supervised dictionary -1=='all', >0=='#unique dict' ")
    group.add_argument("--iteration", type=int, default=10, help="Number of self training iterations")
    group.add_argument("--normalize_embeddings", type=str, default="renorm,center,renorm", help="Normalize embeddings before training")
    group.add_argument("--batch_size", type=int, default=128, help="Batch size")
    group.add_argument("--autoenc_epochs", type=int, default=25, help="Number of initial autoencoder epochs")
    group.add_argument("--autoenc_epoch_size", type=int, default=100000, help="Number of embeddings to be considered in autoencoder training")
    group.add_argument("--mapper_epochs", type=int, default=100, help="Number of initial autoencoder epochs")
    group.add_argument("--csls_knn", type=int, default=10, help="Number of nearest neighbors to consider in CSLS")
    
    group.add_argument("--nonlinear_mapper", type=bool, default=True, help="Use non linear mapper")
    group.add_argument("--nonlinear_autoenc", type=bool, default=True, help="Use non linear Autoencoder")
    
    #Optimizers
    group.add_argument("--map_optimizer", type=str, default="sgd,lr=0.0001", help="Mapping optimizer")
    group.add_argument("--autoenc_optimizer", type=str, default="adam,lr=0.0001", help="Autoencoder optimizer")
    
    #dictionary
    group.add_argument("--dico_max_rank", type=int, default=30000, help="Maximum dictionary words rank (0 to disable)")
    group.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    group.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
    group.add_argument("--induced_dico_c", type=int, default=2000, help="keep dico entry ratio from induced dictionary")

    #loss lambdas
    group.add_argument("--mapping_lambda", type=float, default=1.0, help="Nonlinear Mapper dropout")
    group.add_argument("--cycle_lambda", type=float, default=1.0, help="Nonlinear Mapper dropout")
    group.add_argument("--reconstruction_lambda", type=float, default=1.0, help="Nonlinear Mapper dropout")
    
 
def logistics_params(parser):
    group = parser.add_argument_group('Logistics params.')
    group.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    group.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    group.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    group.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    group.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    group.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
    group.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
    
    

def load_args():
    
    parser = argparse.ArgumentParser("LNMap Training.")
    
    data_params(parser)
    autoenc_params(parser)
    mapping_params(parser)
    training_params(parser)
    logistics_params(parser)

    params = parser.parse_args()
    params = params

    # check parameters
    assert os.path.isfile(params.src_emb)
    assert os.path.isfile(params.tgt_emb)
    
    return params



