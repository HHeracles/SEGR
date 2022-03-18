import torch
import torch.nn as nn
from fastai.vision import *

from modules.model import Model, _default_tfmer_cfg

class BaseAlignment(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_alignment_d_model, _default_tfmer_cfg['d_model'])

        self.loss_weight = ifnone(config.model_alignment_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, self.charset.num_classes)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
            l_lengths: (N,)
            v_lengths: (N,)
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight':self.loss_weight,
                'name': 'alignment'}
                
class SEFusionBlock(nn.Module):
    #def __init__(self, in_planes, planes, stride=1):
    def __init__(self, planes, stride=1):
        super(SEFusionBlock, self).__init__()

        # SE layers
        self.planes = planes
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)
    def forward(self, x):
    #def forward(self, l_feature, v_feature):

        dim = x.dim()
        #if dim==3:
        #    x = x.permute(1, 0, 2)
        #    x=x.unsqueeze(-1)
            
        x=x.unsqueeze(-1)
        out = x

        w = F.avg_pool2d(out, (out.size(2), out.size(3)))

        w = F.relu(self.fc1(w))

        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!    

        output = out + x
        output = F.relu(output)#.contiguous()
        
        if dim==3:
            output = output.squeeze(-1)
        return w, out, output,
        
class SEFusion(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_alignment_d_model, _default_tfmer_cfg['d_model'])

        self.loss_weight = ifnone(config.model_alignment_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, self.charset.num_classes)
        
        self.SEBlock_V = SEFusionBlock(self.max_length)
        self.SEBlock_L = SEFusionBlock(self.max_length)
    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
            l_lengths: (N,)
            v_lengths: (N,)
        """
        weight_l, sum_oput_v, l_feature = self.SEBlock_L(l_feature)
        weight_v, sum_out_v, v_feature = self.SEBlock_V(v_feature)

        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight':self.loss_weight,
                'name': 'alignment'}
                
class SEGateCrossFusion(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_alignment_d_model, _default_tfmer_cfg['d_model'])

        self.loss_weight = ifnone(config.model_alignment_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, self.charset.num_classes)
        
        self.SEBlock_V = SEFusionBlock(self.max_length)
        self.SEBlock_L = SEFusionBlock(self.max_length)
    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
            l_lengths: (N,)
            v_lengths: (N,)
        """
        dim = l_feature.dim()
        weight_l, sum_oput_v, l_SeFeature = self.SEBlock_L(l_feature)
        weight_v, sum_out_v, v_SeFeature = self.SEBlock_V(v_feature)
        
        l_feature, v_feature = l_feature.unsqueeze(-1), v_feature.unsqueeze(-1)
        l_feature_ori, v_feature_ori = l_feature, v_feature

        l_feature = l_feature * weight_v
        l_feature = l_feature_ori + l_feature
        l_feature = F.relu(l_feature)#.contiguous()
        if dim==3:
            l_feature = l_feature.squeeze(-1)
            
        v_feature = v_feature * weight_l
        v_feature = v_feature_ori + v_feature
        v_feature = F.relu(v_feature)#.contiguous()
        if dim==3:
            v_feature = v_feature.squeeze(-1)      

        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight':self.loss_weight,
                'name': 'alignment'}


class SemanticFusionAlignment(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_alignment_d_model, _default_tfmer_cfg['d_model'])

        self.loss_weight = ifnone(config.model_alignment_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, self.charset.num_classes)
        
        self.w_f = nn.Linear(2 * d_model, d_model)
        self.w_i = nn.Linear(2 * d_model, d_model)
        self.w_c = nn.Linear(2 * d_model, d_model)
        self.w_o = nn.Linear(2 * d_model, d_model)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
            l_lengths: (N,)
            v_lengths: (N,)
            l_featrue as input and cell, v_feaature as hinden 
        """
        i , c, h = l_featrue, l_featrue, v_feaature
        # forgate gate
        t = self.w_f(torch.cat((l_feature, v_feature), dim=2))
        ft = torch.sigmoid(t)
        
        #input gate
        it = torch.sigmoid(self.w_i(torch.cat((l_feature, v_feature), dim=2)))
        ct_ = torch.tanh(self.w_c(torch.cat((l_feature, v_feature), dim=2)))
        ct = ft * c + it * ct_
        
        #output gate
        ot = torch.sigmoid(self.w_o(torch.cat((l_feature, v_feature), dim=2)))
        ht = ot * torch.tanh(ct)
        
        
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight':self.loss_weight,
                'name': 'alignment'}
                
                                
class VotingFusion(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_alignment_d_model, _default_tfmer_cfg['d_model'])

        self.loss_weight = ifnone(config.model_alignment_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, self.charset.num_classes)

    def forward(self, res):

        vot_res = res

        cls_1 = F.softmax(vot_res[0]['logits'], dim=2)
        cls_2 = F.softmax(vot_res[1]['logits'], dim=2)
        cls_3 = F.softmax(vot_res[2]['logits'], dim=2)
        fusion_logits = vot_res[0]['logits']

        for i, cl in enumerate(cls_1):
            score_1 = cl.max(dim=1)[0]

            score_2 = cls_2[i].max(dim=1)[0]
            score_3 = cls_3[i].max(dim=1)[0]

            for j in range(score_1.shape[0]):
                if score_1[j] < score_2[j]:
                    fusion_logits[i,j,:] = vot_res[1]['logits'][i,j,:]
                    if score_2[j] < score_3[j]:
                        fusion_logits[i,j,:] = vot_res[2]['logits'][i,j,:]
                else:
                    fusion_logits[i,j,:] = vot_res[0]['logits'][i,j,:]
                    if score_1[j] < score_3[j]:
                        fusion_logits[i,j,:] = vot_res[2]['logits'][i,j,:]

        pt_lengths = self._get_length(fusion_logits)
        res =  {'logits': fusion_logits, 'pt_lengths': pt_lengths, 'loss_weight':1.0,'name': 'voting'}
        return res


