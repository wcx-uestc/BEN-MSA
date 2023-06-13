import torch
import torch.nn as nn
from torch.nn import Parameter
from modules.attention.transformer import TransformerEncoder
from modules.InfoNCE import *


# 不加cls 不更新bottleneck效果最佳

class BottleNeckFusion(nn.Module):
    def __init__(self, config):
        super(BottleNeckFusion, self).__init__()
        self.embed_dim = config.embed_dim
        self.recursion = config.recursion
        self.num_heads = config.num_heads
        self.output_method = config.output_method
        self.num_self_attn = config.num_self_attn
        self.info_nce = InfoNCE(config)
        # TODO to embed_dim
        self.audio_linear = nn.Linear(74, self.embed_dim)
        self.visual_linear = nn.Linear(35, self.embed_dim)
        self.text_linear = nn.Linear(768, self.embed_dim)
        # TODO self-attention
        self.audio_transformer = TransformerEncoder(embed_dim=32, num_heads=8, layers=self.num_self_attn, attn_mask=False)
        self.visual_transformer = TransformerEncoder(embed_dim=32, num_heads=8, layers=self.num_self_attn, attn_mask=False)
        # TODO fuse
        # self.fc_a = nn.Linear(self.embed_dim, 1)
        # self.fc_v = nn.Linear(self.embed_dim, 1)
        # self.fc_t = nn.Linear(self.embed_dim, 1)
        self.fc = nn.Linear(self.embed_dim, 1)


    def forward(self, visual, audio, text, bottleneck):
        # TODO self-attention
        # [seq_len, batch_size, emb]
        audio, visual = self.audio_linear(audio).permute(1, 0, 2), self.visual_linear(visual).permute(1, 0, 2)
        a_in, v_in, t_in = self.audio_transformer(audio), self.visual_transformer(visual), self.text_linear(text)
        # [batch_size, seq_len, emb]
        a_fea, v_fea, t_fea = a_in.transpose(0, 1), v_in.transpose(0, 1), t_in
        # TODO InfoNCE Loss
        loss_at = self.info_nce(t_fea, a_fea)
        loss_vt = self.info_nce(t_fea, v_fea)
        # [batch_size, emb]
        a_fea = torch.mean(a_fea, dim=1)
        v_fea = torch.mean(v_fea, dim=1)
        t_fea = torch.mean(t_fea, dim=1)
        a_fea, v_fea, t_fea = self.fc(a_fea), self.fc(v_fea), self.fc(t_fea)
        result = (a_fea + v_fea + t_fea) / 3
        # result = torch.concat((a_out, v_out, t_out), dim=1)
        # result = self.fc(result)
        # a_out = torch.mm(a_out, torch.transpose(self.fc.weight[:, :32], 0, 1)) + self.fc.bias / 3
        # v_out = torch.mm(v_out, torch.transpose(self.fc.weight[:, 32:64], 0, 1)) + self.fc.bias / 3
        # t_out = torch.mm(t_out, torch.transpose(self.fc.weight[:, 64:], 0, 1)) + self.fc.bias / 3
        # a_fea, v_fea, t_fea = self.fc_a(a_out), self.fc_v(v_out), self.fc_t(t_out)
        # result = (a_fea + v_fea + t_fea) / 3
        # a_out = torch.mm(a_out, torch.transpose(self.fc_a.weight, 0, 1)) + self.fc_a.bias / 3
        # v_out = torch.mm(v_out, torch.transpose(self.fc_v.weight, 0, 1)) + self.fc_v.bias / 3
        # t_out = torch.mm(t_out, torch.transpose(self.fc_t.weight, 0, 1)) + self.fc_t.bias / 3
        return a_fea, v_fea, t_fea, result, loss_at, loss_vt
