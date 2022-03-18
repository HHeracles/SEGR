import torch
import torch.nn as nn
from fastai.vision import *

from .model_vision import BaseVision, PositionAttentionKeyAndValueVision, BasePositionVision
from .model_language import BCNLanguage, BCNVisionLanguage, BCNInstanceNorm1dLanguage, BCNInstanceNorm1dAndConv1dLanguage,BCNInstanceNorm1dAndConv1dGateFusionLanguage,BCNInstanceMemoryAndNorm1dGateFusionLanguage,BCNFusionVisionAndLanguage,BCNIgnoreMaskLanguage,BCNSoftMapLanguage,BCNLanguagelinearPool, BCNMeshLanguage,BCNBLSTMLanguage,BCNSELanguage,BCNSEANDNormalLanguage,BCNSEMixtureLanguage,BCNOnlySEValuesLanguage,BCNOnlySEKeyesLanguage,BCNVisionAndSELanguage
from .model_recognition import RecognitionLanguage
from .model_alignment import BaseAlignment, SEFusion, SEGateCrossFusion, SemanticFusionAlignment
from .Rs_GCN import GraphyReason

############################################
#### This model contain GCNReason module ###
############################################
class SEGRModel_1220(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        self.graphyReason = GraphyReason(config)

    def forward(self, images, *args):
        v_res = self.vision(images)
        a_res = v_res
        all_l_res, all_a_res,all_g_res = [], [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.max_length)  # TODO:move to langauge model

            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
            ###############################
            #####     graphy feature   ####
            ###############################
            gcn_res = self.graphyReason(a_res['feature'])
            all_g_res.append(gcn_res)

        if self.training:
            return all_g_res, all_a_res, all_l_res, v_res
        else:
            return gcn_res, a_res, all_l_res[-1], v_res   
              
############################################################################
#### This model contain SEBlock and GCNReason,                           ###
############################################################################
class SEGRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseVision(config)
        self.language = BCNSELanguage(config)
        self.alignment = BaseAlignment(config)
        self.graphyReason = GraphyReason(config)

    def forward(self, images, *args):
        v_res = self.vision(images)
        gcn_res = v_res
        all_l_res, all_a_res,all_g_res, all_v_res = [], [], [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(gcn_res['logits'], dim=-1)
            lengths = gcn_res['pt_lengths']
            lengths.clamp_(2, self.max_length)  # TODO:move to langauge model

            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
                
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
            ###############################
            #####     graphy feature   ####
            ###############################
            gcn_res = self.graphyReason(a_res['feature'])
            all_g_res.append(gcn_res)

        if self.training:
            return all_g_res, all_a_res, all_l_res, v_res
        else:
            return gcn_res, a_res, all_l_res[-1], v_res   

############################################
### the Only GCN of  ###
############################################
class SEGRModel_onlygcn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        self.graphyReason = GraphyReason(config)

    def forward(self, images, *args):
        v_res = self.vision(images)
        gcn_res = v_res
        all_l_res, all_a_res,all_g_res = [], [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(gcn_res['logits'], dim=-1)
            lengths = gcn_res['pt_lengths']
            lengths.clamp_(2, self.max_length)  # TODO:move to langauge model
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
            ###############################
            #####     graphy feature   ####
            ###############################
            gcn_res = self.graphyReason(a_res['feature'])
            all_g_res.append(gcn_res)

        if self.training:
            return all_g_res, all_a_res, all_l_res, v_res
        else:
            return gcn_res, a_res, all_l_res[-1], v_res   
                                        

