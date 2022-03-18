import logging
import torch.nn as nn
from fastai.vision import *
import copy

from modules.model import _default_tfmer_cfg
from modules.model import Model
from modules.embedings import Embeddings
from .model_alignment import BaseAlignment
from modules.transformer import (PositionalEncoding,
                                 TransformerDecoder,
                                 TransformerDecoderLayer,
                                 TransformerInstanceNorm1dDecoderLayer,
                                 TransformerInstanceNorm1dAndConv1dDecoderLayer,
                                 TransformerInstanceMemroyAndNorm1dDecoderLayer,
                                 TransformerGateFusionDecoder,
                                 TransformerDecoderVisionLayer,
                                 TransformerFusionVisionAndLanguageDecoderLayer,
                                 TransformerGateFusionVisionAndLanguageDecoder,
                                 TransformerAttMapDecoderLayer,
                                 TransformerAttMapDecoder,
                                 TransformerLinearPoolDecoderLayer,
                                 TransformerLanguageRecogLayer,
                                 TransformerMeshDecoder,
                                 TransformerBLSTMDecoderLayer,
                                 TransformerSEBlockDecoderLayer,
                                 TransformerSEANDINormalDecoderLayer,
                                 TransformerSEMixtureANDINormalDecoderLayer,
                                 TransformerOnlySEValuesBlockDecoderLayer,
                                 TransformerOnlySEKeyesBlockDecoderLayer)
from .model_recognition import MultiLevelLanguageRecog
class BCNLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res

class BCNInstanceNorm1dLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerInstanceNorm1dDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """

        if self.detach: tokens = tokens.detach()

        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)
        
        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}

        return res
        

class BCNInstanceNorm1dAndConv1dLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerInstanceNorm1dAndConv1dDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint,strict = None)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """

        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)
        
        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}

        return res
        
class BCNInstanceNorm1dAndConv1dGateFusionLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        alignment_layer = BaseAlignment(config)
        decoder_layer = TransformerInstanceNorm1dDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerGateFusionDecoder(decoder_layer, alignment_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint,strict = None)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """

        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)
        
        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)
        
        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}

        return res


class BCNFusionVisionAndLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        alignment_layer = BaseAlignment(config)

        decoder_layer = TransformerFusionVisionAndLanguageDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)

        self.model = TransformerGateFusionVisionAndLanguageDecoder(decoder_layer, alignment_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint,strict = None)

    def forward(self, vision_feature, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """

        if self.detach: tokens = tokens.detach()
        language = self.proj(tokens)  # (N, T, E)
        language = language.permute(1, 0, 2)  # (T, N, E)
        language = self.token_encoder(language)  # (T, N, E)

        padding_mask = self._get_padding_mask(lengths, self.max_length)
        
        zeros = language.new_zeros(*language.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, vision_feature, language,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)
        
        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}

        return res

        
class BCNInstanceMemoryAndNorm1dGateFusionLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        alignment_layer = BaseAlignment(config)

        decoder_layer = TransformerInstanceMemroyAndNorm1dDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerGateFusionDecoder(decoder_layer, alignment_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint,strict = None)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """

        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)

        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)

        padding_mask = self._get_padding_mask(lengths, self.max_length)
        
        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)
        
        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}

        return res
############################
####   model A  ############
############################
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

class CANLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout,
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res

############################
####   model B  ############
############################
class BCNVisionLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        
        in_channels = ifnone(config.model_language_in_channels, 512)
        num_channels = ifnone(config.model_language_num_channels, 64)
        mode = ifnone(config.model_vision_attention_mode, 'nearest')
        h=8 
        w=32
        c = copy.deepcopy
        self.vision_feature_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.vision_feature_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(h, w), mode=mode)
        )
        
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decodervision_layer = TransformerDecoderVisionLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decodervision_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint, strict=None)

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
        tokens = self.proj(tokens)  # (N, T, E)
        tokens = tokens.permute(1, 0, 2)  # (T, N, E)
        tokens = self.token_encoder(tokens)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = tokens.new_zeros(*tokens.shape)
        query = self.pos_encoder(zeros) + tokens
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        features = []
        for i in range(0, len(self.vision_feature_encoder)):
            vision_feature = self.vision_feature_encoder[i](vision_feature)
            features.append(vision_feature)
        for i in range(0, len(self.vision_feature_decoder) - 1):
            vision_feature = self.vision_feature_decoder[i](vision_feature)
            vision_feature = vision_feature + features[len(self.vision_feature_decoder) - 2 - i]
        vision_feature = self.vision_feature_decoder[-1](vision_feature)
        embed = vision_feature.flatten(2, 3)  

        embed=embed.permute(2,0,1)
        output = self.model(query, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res



class BCNIgnoreMaskLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = None

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res



class BCNSoftMapLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        self.att_map = nn.Parameter(torch.FloatTensor(nhead, self.max_length, self.max_length))

        
        decoder_layer = TransformerAttMapDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerAttMapDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.att_map, 0, 1)
        
    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()

        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)# att_mask
        att_map = None
        att_map = self.att_map

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                att_map=att_map,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res

class BCNLanguagelinearPool(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerLinearPoolDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)# att_mask

        #location_mask = None
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                #memory_mask=padding_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res

class BCNMeshLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        
        encoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.encoder = TransformerMeshDecoder(encoder_layer, num_layers)
        
        decoder_layer = TransformerLanguageRecogLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.languageLayer = MultiLevelLanguageRecog(decoder_layer, num_layers, num_layers)
                
        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

        d_ff = 4*d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        encoder_output,encoder_outputs = self.encoder(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        encoder_output = encoder_output.permute(1, 0, 2)  # (N, T, E)

        # decoder
        decoder_output = self.languageLayer(qeury, encoder_outputs, encoder_outputs,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)
                
        decoder_output_pw = self.fc2(self.dropout_2(F.relu(self.fc1(decoder_output))))
        decoder_output_pw = self.dropout(decoder_output_pw)
        decoder_output_pw = self.layer_norm(decoder_output + decoder_output_pw)
        
        decoder_output = decoder_output.permute(1, 0, 2)  # (N, T, E)
        logits = self.cls(decoder_output)  # (N, T, C)
        
        pt_lengths = self._get_length(logits)

        res =  {'feature': decoder_output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}

        return res
class BCNBLSTMLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerBLSTMDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()

        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)
        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res
class BCNSEANDNormalLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decoder_layer = TransformerSEANDINormalDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()

        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res
class BCNSELanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decoder_layer = TransformerSEBlockDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()


        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res

class BCNOnlySEKeyesLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decoder_layer = TransformerOnlySEKeyesBlockDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()

        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)
        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res
class BCNOnlySEValuesLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decoder_layer = TransformerOnlySEValuesBlockDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()

        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res        
class BCNSEMixtureLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decoder_layer = TransformerSEMixtureANDINormalDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()

        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)

        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res
class BCNVisionAndSELanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decoder_layer = TransformerSEBlockDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug, max_length=2*self.max_length)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, vision_feature, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()

        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)
        padding_mask = None

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        location_mask = None

        vision_feature = vision_feature.permute(1, 0, 2)
        embed = torch.cat([embed, vision_feature],dim=0)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)

        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res