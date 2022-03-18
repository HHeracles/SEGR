import torch
from torch import nn
from torch.nn import functional as F
from fastai.vision import *
from modules.model import Model, _default_tfmer_cfg

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class Rs_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None


        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)
        R = torch.matmul(theta_v, phi_v)
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        v_star = W_y + v

        return v_star
from torch.nn import LayerNorm
class GraphyReason(Model):
    ##############################
    ##### Graphy Seem     ########
    ##############################
    def __init__(self, config):
        super(GraphyReason, self).__init__(config)
        d_model = ifnone(config.model_alignment_d_model, _default_tfmer_cfg['d_model'])
        self.loss_weight = ifnone(config.model_alignment_loss_weight, 1.0)
        
        #self.cfg = config.clone()
        #self.embed_size = cfg.MODEL.TSR.EMBED_SIZE
        self.embed_size = 512
        self.Rs_GCN_1 = Rs_GCN(in_channels=self.embed_size, inter_channels=self.embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=self.embed_size, inter_channels=self.embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=self.embed_size, inter_channels=self.embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=self.embed_size, inter_channels=self.embed_size)
        
        self.fc = nn.Linear(512, self.embed_size)
        self.cls = nn.Linear(d_model, self.charset.num_classes)

    def forward(self, feature,decoder_targets=None, word_targets=None):
        """Extract image feature vectors."""
        feature = l2norm(feature)
        #feature = self.fc(feature)

        # GCN reasoning
        # -> B,D,N
        GCN_feature = feature.permute(0, 2, 1)
        GCN_feature = self.Rs_GCN_1(GCN_feature)

        GCN_feature = self.Rs_GCN_2(GCN_feature)

        GCN_feature = self.Rs_GCN_3(GCN_feature)

        GCN_feature = self.Rs_GCN_4(GCN_feature)
        # -> B,N,D
        GCN_feature = GCN_feature.permute(0, 2, 1)

        GCN_feature = l2norm(GCN_feature)
        GCN_feature = GCN_feature + feature
        
        gcn_logits = self.cls(GCN_feature)  # (N, T, C)
        gcn_pt_lengths = self._get_length(gcn_logits)

        return {'feature': GCN_feature,'logits': gcn_logits, 'pt_lengths': gcn_pt_lengths, 'loss_weight':self.loss_weight,
                'name': 'grahpyReason'}
        
