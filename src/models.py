import torch
from torch import nn

from .utils import load_embeddings, normalize_embeddings


class NonLinearMapper(nn.Module):
    """
    Single hidden layer FFN with tanh activation
    """
    def __init__(self, params, mapper_AB=True):
        super(NonLinearMapper, self).__init__()
        self.emb_dim_A = params.emb_dim_autoenc_A if mapper_AB else params.emb_dim_autoenc_B
        self.emb_dim_B = params.emb_dim_autoenc_B if mapper_AB else params.emb_dim_autoenc_A
        self.hidden_dim = params.mapper_hidden_dim

        self.lin_layer1 = nn.Linear(self.emb_dim_A, self.hidden_dim)
        self.lin_layer2 = nn.Linear(self.hidden_dim, self.emb_dim_B)

        if getattr(params, 'map_id_init', True):
            nn.init.zeros_(self.lin_layer1.weight.data).fill_diagonal_(1)
            nn.init.zeros_(self.lin_layer2.weight.data).fill_diagonal_(1)

        self.dropuout_layer = nn.Dropout(params.mapper_dropout)
        self.nonlinearity = nn.Tanh()

    def forward(self, x):
        x = self.nonlinearity(self.dropuout_layer(self.lin_layer1(x)))
        x = self.lin_layer2(x)
        return x


class Encoder(nn.Module):
    """
    3 layer FFN
    """
    def __init__(self, params, lang_A=True):
        super(Encoder, self).__init__()

        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        if lang_A:
            self.bottleneck_dim = params.emb_dim_autoenc_A
        else:
            self.bottleneck_dim = params.emb_dim_autoenc_B

        self.lin1 = nn.Linear(self.emb_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, self.bottleneck_dim)

        self.dropuout_layer = nn.Dropout(0.25)
        self.nonlinearity = nn.PReLU()

    def forward(self, x):
        x = self.nonlinearity(self.dropuout_layer(self.lin1(x)))
        x = self.nonlinearity(self.dropuout_layer(self.lin2(x)))
        x = self.lin3(x)
        return x


class Decoder(nn.Module):
    """
    3 layer FFN
    """
    def __init__(self, params, lang_A=True):
        super(Decoder, self).__init__()

        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        if lang_A:
            self.bottleneck_dim = params.emb_dim_autoenc_A
        else:
            self.bottleneck_dim = params.emb_dim_autoenc_B

        self.lin1 = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, self.emb_dim)

        self.dropuout_layer = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.nonlinearity = nn.PReLU()

    def forward(self, x):
        x = self.nonlinearity(self.dropuout_layer(self.lin1(x)))
        x = self.nonlinearity(self.dropuout_layer(self.lin2(x)))
        x = self.tanh(self.lin3(x))
        return x  


class LinEncoder(nn.Module):
    def __init__(self, params, lang_A=True):
        super(LinEncoder, self).__init__()
        print("Linear Encoder")
        self.emb_dim = params.emb_dim
        if lang_A:
            self.bottleneck_dim = params.emb_dim_autoenc_A
        else:
            self.bottleneck_dim = params.emb_dim_autoenc_B      
        self.encoder = nn.Linear(self.emb_dim, self.bottleneck_dim)

    def forward(self, x):
        return self.encoder(x)

class LinDecoder(nn.Module):
    def __init__(self, params, lang_A=True):
        super(LinDecoder, self).__init__()
        print("Linear Decoder")
        self.emb_dim = params.emb_dim
        if lang_A:
            self.bottleneck_dim = params.emb_dim_autoenc_A
        else:
            self.bottleneck_dim = params.emb_dim_autoenc_B
        self.decoder = nn.Linear(self.bottleneck_dim, self.emb_dim)
    def forward(self, x):
        return self.decoder(x)

def build_model(params):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    tgt_dico, _tgt_emb = load_embeddings(params, source=False)
    params.tgt_dico = tgt_dico
    tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
    tgt_emb.weight.data.copy_(_tgt_emb)

    # autoencoder
    encoder_A = Encoder(params, lang_A=True) if params.nonlinear_autoenc else LinEncoder(params, lang_A=True)
    decoder_A = Decoder(params, lang_A=True) if params.nonlinear_autoenc else LinDecoder(params, lang_A=True)
    encoder_B = Encoder(params, lang_A=False) if params.nonlinear_autoenc else LinEncoder(params, lang_A=False)
    decoder_B = Decoder(params, lang_A=False) if params.nonlinear_autoenc else LinDecoder(params, lang_A=False)

    # mappers
    if params.nonlinear_mapper == False:
        print("*******Linear Mapper!")
        mapping_G = nn.Linear(params.emb_dim_autoenc_A,
                              params.emb_dim_autoenc_B, bias=False)
        mapping_F = nn.Linear(params.emb_dim_autoenc_B,
                              params.emb_dim_autoenc_A, bias=False)
        if getattr(params, 'map_id_init', True):
            nn.init.zeros_(mapping_G.weight.data).fill_diagonal_(1)
            nn.init.zeros_(mapping_F.weight.data).fill_diagonal_(1)
            # mapping_G.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
            # mapping_F.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
    else:
        mapping_G = NonLinearMapper(params, mapper_AB=True)
        mapping_F = NonLinearMapper(params, mapper_AB=False)

    # cuda
    if params.cuda:
        src_emb.cuda()
        tgt_emb.cuda()
        encoder_A.cuda()
        decoder_A.cuda()
        encoder_B.cuda()
        decoder_B.cuda()
        mapping_G.cuda()
        mapping_F.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping_G, mapping_F, encoder_A, decoder_A, encoder_B, decoder_B



