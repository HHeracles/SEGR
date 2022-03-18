import torch
import torch.nn as nn
import torch.nn.functional as F
#from lib.config import cfg
#import lib.utils as utils
from layers.basic_att import BasicAtt

class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        #print("sc_att_0_self.attention_basic:::",self.attention_basic)
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)
        #print("att_map.shape:",att_map.shape)
        #att_mask = None
        if att_mask is not None:
            #att_mask = att_mask.unsqueeze(1)
            #print("att_map.shape_2:",att_map.shape)
            #print("att_mask.shape:",att_mask.shape)
            att_mask_ext = att_mask.unsqueeze(-1)
            #print("att_mask_ext.shape:",att_mask_ext.shape)
            #print("att_mask_ext:",att_mask_ext)
            #print("torch.sum(att_mask_ext, -2):",torch.sum(att_mask_ext, -2))
            #input()
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
            #att_map_pool = torch.sum(att_map * att_mask, -2) / torch.sum(att_mask, -2)
        else:
            att_map_pool = att_map.mean(-2)
        #print("att_map_pool.shape:",att_map_pool.shape)
        
        alpha_spatial = self.attention_last(att_map)
        #print("alpha_spatial.shape_0:",alpha_spatial.shape)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)

        alpha_spatial = alpha_spatial.squeeze(-1)
        #print("att_mask.shape:",att_mask.shape)
        #print("att_mask:",att_mask)
        #print("alpha_spatial_1:",alpha_spatial)
        
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        #print("alpha_spatial_2:",alpha_spatial)
        #input()
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)
        print("alpha_spatial.shape:",alpha_spatial.shape)
        attn = value1 * value2 * alpha_channel
        #print("sc_att_attn.shape:",attn.shape)
        #input()
        return attn, alpha_spatial
