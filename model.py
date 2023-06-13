import torch
from modules.vanilla_fusion import BottleNeckFusion
from modules.encoder.text import TextEncoder
# from modules.fusion import BottleNeckFusion
from torch import nn


class Multi_models(nn.Module):
    def __init__(self, config):
        super(Multi_models, self).__init__()
        self.seq = config.seq
        self.embed_dim = config.embed_dim
        self.text_enc = TextEncoder(config)
        if config.frozen:
            self.text_enc.requires_grad_(False)
        self.bottleneck = nn.Parameter((torch.Tensor(1, self.seq, self.embed_dim)))
        nn.init.xavier_uniform_(self.bottleneck)
        self.BottleNeckFuse = BottleNeckFusion(config)

    def forward(self, visual, audio, bert_sentences, bert_sentence_types, bert_sentence_att_mask):
        t_in = self.text_enc(bert_sentences, bert_sentence_types, bert_sentence_att_mask)
        a_result, v_result, t_result, result, loss_at, loss_vt = self.BottleNeckFuse(visual, audio, t_in, self.bottleneck)
        return a_result, v_result, t_result, result, loss_at, loss_vt


