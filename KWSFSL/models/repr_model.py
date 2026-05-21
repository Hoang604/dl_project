import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from models.utils import register_model
from models.encoder.DSCNN import DSCNNL, DSCNNM, DSCNNS, DSCNNL_NONORM, \
                DSCNNL_LAYERNORM, DSCNNM_LAYERNORM, DSCNNS_NONORM, DSCNNS_LAYERNORM, DSCNNL_BATCHNORM, DSCNNM_BATCHNORM, DSCNNS_BATCHNORM
from models.encoder.ViT import ViTEncoder

from models.preprocessing import MFCC

from models.losses.triplet import online_triplet_loss



class ReprModel(nn.Module):
    def __init__(self, encoder, preprocessing, 
            criterion, x_dim, emb_norm, feat_extractor=False):
        super(ReprModel, self).__init__()
        self.encoder = encoder
        self.preprocessing = preprocessing
        self.emb_norm = emb_norm

        # get embedding size
        x_fake = torch.Tensor(1,x_dim[0],x_dim[1],x_dim[2] )
        z = self.encoder.forward(x_fake)
        z_dim = z.size(1)

        #setup loss
        if criterion['type'] == 'triplet':
            self.criterion = online_triplet_loss(criterion)
        else:
            raise ValueError("Loss {} not supported in best mechanism".format(criterion['type']))

        self.feat_extractor = feat_extractor
    
    def get_embeddings(self, x):
        # x is a batch of data
        if self.preprocessing:
            x = self.preprocessing.extract_features(x)
        if self.feat_extractor:
            zq = zq 
        zq = self.encoder.forward(x)
        if self.emb_norm:
            zq = F.normalize(zq, p=2.0, dim=-1)
        return zq

    def loss(self, x):
        # get information
        n_class = x.size(0)
        n_sample = x.size(1)

        #  inference
        x = x.view(n_class * n_sample, *x.size()[2:]).cuda()
        zq = self.get_embeddings(x)
        
        # loss
        loss_val = self.criterion.compute(zq, n_sample, n_class)

        return loss_val, {
            'loss': loss_val.item(),
        }

    def loss_class(self, x, labels):
        zq = self.get_embeddings(x)
        return self.criterion.compute(zq, labels)



def get_encoder(encoding, x_dim, hid_dim, out_dim, **kwargs):
    if encoding == 'DSCNNL':
        return DSCNNL(x_dim)
    elif encoding == 'DSCNNL_NONORM':
        return DSCNNL_NONORM(x_dim)
    elif encoding == 'DSCNNL_LAYERNORM':
        return DSCNNL_LAYERNORM(x_dim)        
    elif encoding == 'DSCNNL_BATCHNORM':
        return DSCNNL_BATCHNORM(x_dim)    
    elif encoding == 'DSCNNM':
        return DSCNNM(x_dim)
    elif encoding == 'DSCNNM_LAYERNORM':
        return DSCNNM_LAYERNORM(x_dim) 
    elif encoding == 'DSCNNM_BATCHNORM':
        return DSCNNM_BATCHNORM(x_dim)      
    elif encoding == 'DSCNNS':
        return DSCNNS(x_dim)
    elif encoding == 'DSCNNS_NONORM':
        return DSCNNS_NONORM(x_dim)
    elif encoding == 'DSCNNS_LAYERNORM':
        return DSCNNS_LAYERNORM(x_dim)    
    elif encoding == 'DSCNNS_BATCHNORM':
        return DSCNNS_BATCHNORM(x_dim)    
    elif encoding == 'ViT':
        patch_size_str = kwargs.get('patch_size', '7,2')
        if isinstance(patch_size_str, str):
            patch_size = tuple(map(int, patch_size_str.split(',')))
        else:
            patch_size = patch_size_str
            
        num_heads = kwargs.get('num_heads', 4)
        num_layers = kwargs.get('num_layers', 4)
        dropout = kwargs.get('dropout', 0.1)
        
        return ViTEncoder(
            x_dim, hid_dim, out_dim, 
            patch_size=patch_size, 
            num_heads=num_heads, 
            num_layers=num_layers, 
            dropout=dropout
        )
    else:
        raise ValueError("Model {} is not valid".format(encoding))


@register_model('repr_conv')
def load_repr_conv(**kwargs):
    z_norm = kwargs['z_norm']
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    encoding = kwargs['encoding']
    print(encoding, x_dim, hid_dim, z_dim)

    #get encoder
    encoder_kwargs = kwargs.copy()
    for k in ['x_dim', 'hid_dim', 'z_dim', 'encoding']:
        encoder_kwargs.pop(k, None)
    encoder = get_encoder(encoding, x_dim, hid_dim, z_dim, **encoder_kwargs)

    # get preprocessing
    preprocessing = False
    if 'mfcc' in kwargs.keys():
        audio_prep = kwargs['mfcc']
        preprocessing = MFCC(audio_prep)

    #get criterion 
    criterion = kwargs['loss'] if 'loss' in kwargs.keys() else False

    # get feat_extractor stage, e.g. wav2vec
    feat_extractor = False

    return ReprModel(encoder, preprocessing, criterion, x_dim, z_norm, feat_extractor)
