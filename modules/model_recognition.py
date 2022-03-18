import logging
import torch.nn as nn
from fastai.vision import *
import copy
from torch.nn import ModuleList

from modules.model import _default_tfmer_cfg
from modules.model import Model
from modules.embedings import Embeddings

from modules.transformer import (PositionalEncoding, 
                                 TransformerDecoder,
                                 TransformerDecoderLayer,
                                 TransformerVisionRecogLayer,
                                 TransformerLanguageRecogLayer)

def get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def memory_encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def memory_decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor, 
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))
                             
class MultiLevelVisionRecog(nn.Module):
    def __init__(self, decoder_layer, vision_num_layers, padding_idx=float('-inf'), norm=None, dropout=0.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,in_channels=512, num_channels=64, 
                 h=8, w=32, mode='nearest'):
        super(MultiLevelVisionRecog, self).__init__()

        self.layers =  get_clones(decoder_layer, vision_num_layers)
        self.vision_num_layers = vision_num_layers
        self.norm = norm
        self.padding_idx = padding_idx
        
        self.k_encoder = nn.Sequential(
            memory_encoder_layer(in_channels, num_channels, s=(1, 2)),
            memory_encoder_layer(num_channels, num_channels, s=(2, 2)),
            memory_encoder_layer(num_channels, num_channels, s=(2, 2)),
            memory_encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder = nn.Sequential(
            memory_decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            memory_decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            memory_decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            memory_decoder_layer(num_channels, in_channels, size=(h, w), mode=mode)
        )
        #self.fc = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = 512
        self.layer_norm = nn.LayerNorm(self.d_model)
        

    def forward(self, input, attention_weights=None):

        queries,keys,values = input, input, input,
        features=[]
        for i in range(0, len(self.k_encoder)):
            keys = self.k_encoder[i](keys)
            features.append(keys)
        for i in range(0, len(self.k_decoder) - 1):
            keys = self.k_decoder[i](keys)
            keys = keys + features[len(self.k_decoder) - 2 - i]
        keys = self.k_decoder[-1](keys)

        queries = queries.flatten(2, 3).permute(0, 2, 1)
        keys = keys.flatten(2, 3).permute(0, 2, 1)
        values = values.flatten(2, 3).permute(0, 2, 1)
        
        #queries = F.relu(self.fc(queries))

        queries = F.relu(queries)
        queries = self.dropout(queries)
        queries = self.layer_norm(queries)
        
        keys = F.relu(keys)
        keys = self.dropout(keys)
        keys = self.layer_norm(keys)
        
        values = F.relu(values)
        values = self.dropout(values)
        values = self.layer_norm(values)
        
        attention_mask = (torch.sum(queries, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        
        outs = []
        i=0

        for layer in self.layers:
            queries = layer(queries, keys, values, attention_mask=attention_mask, attention_weights=attention_weights)

            print("queries.shape:",queries.shape)
            outs.append(queries.unsqueeze(1))
            print("queries.unsqueeze(1)).shape:",queries.unsqueeze(1).shape)
            i=i+1

        outs = torch.cat(outs, 1)
        return outs, attention_mask

class MultiLevelLanguageRecog(nn.Module):
    def __init__(self, decoder_layer,  vision_num_layers, language_num_layers, padding_idx=float('-inf'), norm=None,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelLanguageRecog, self).__init__()

        self.layers =  get_clones(decoder_layer, language_num_layers)
        self.language_num_layers = language_num_layers
        self.vision_num_layers = vision_num_layers
        self.norm = norm
        
        self.padding_idx = padding_idx
        

        
    def forward(self, query, keys, values, memory2=None, tgt_mask=None,
                memory_mask=None, memory_mask2=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, memory_key_padding_mask2=None):

        outs = []
        out = query

        for layer in self.layers:
            out = layer(out, keys, values, memory2=None, tgt_mask=None,
                memory_mask=None, memory_mask2=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, memory_key_padding_mask2=None)
            outs.append(out.unsqueeze(1))

        return out
            
def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor, 
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True)) 
################################################
####   model C:RecognitionLanguage  ############
################################################
class RecognitionLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        self.d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        vision_num_layers = ifnone(config.model_vision_num_layers, 4)
        language_num_layers = ifnone(config.model_language_num_layers, 4)

        c = copy.deepcopy
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, True)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.fc = nn.Linear(self.d_inner, self.d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        # self.tgt_embed=nn.Sequential(Embeddings(d_model, self.charset.num_classes), c(self.token_encoder))
        vision_recognition_layer = TransformerVisionRecogLayer(d_model, nhead, config, self.d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.visionLayer = MultiLevelVisionRecog(vision_recognition_layer, vision_num_layers)
        
        language_recognition_layer = TransformerLanguageRecogLayer(d_model, nhead, self.d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.languageLayer = MultiLevelLanguageRecog(language_recognition_layer, vision_num_layers, language_num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)
        self.padding_idx = 0
        
        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint, strict = None)

    def forward(self, vision_feature, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        
        #############################################
        ####   model_1  standard transformer    #####
        #############################################
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        #############################################
        ####   model_2  standard transformer    #####
        #############################################
        enc_output, mask_enc  = self.visionLayer(vision_feature)
        output = self.languageLayer(qeury, enc_output, enc_output,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)
        
        output = output.permute(1, 0, 2)  # (N, T, E)
        logits = self.cls(output)  # (N, T, C)
        
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res
    

