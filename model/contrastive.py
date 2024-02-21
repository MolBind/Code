"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from lavis.common.dist_utils import is_dist_avail_and_initialized
from bind import Blip2Base
from pytorch_lightning.utilities import distributed
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    print('running here')
    return output

@torch.no_grad()
def pl_concat_all_gather(tensor):

    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = distributed.gather_all_tensors(tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output


class Model(Blip2Base):

    def __init__(
        self,
        gtm,
        lm,
        bert_name,
        temperature,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        args=None,
        w_gt=0.5,
        w_ct=0.5,
        w_molpro=0.5
    ):
        super().__init__()
        self.gtm = gtm
        self.lm = lm
        self.args = args
        self.tokenizer = self.init_tokenizer()

        self.w_gt = w_gt
        self.w_ct = w_ct
    
        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.conf_encoder, self.ln_conf, self.dictionary_mol = self.init_unimol_mol_encoder(args)
        self.pro_encoder, self.ln_pro, self.dictionary_pro = self.init_unimol_pro_encoder(args)


        self.Qformer = self.init_bert_encoder(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
    
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.graph_proj = nn.Linear(gin_hidden_dim, embed_dim)
        
        self.conf_proj = nn.Linear(args.unimol_encoder_embed_dim, embed_dim)
        self.pro_proj = nn.Linear(args.unimol_encoder_embed_dim, embed_dim)

        self.temperature_gt = temperature
        self.temperature_ct = temperature
        self.temperature_molpro = temperature

    def contrastive(self, features_graph, features_text2d, feature_conf, feature_text3d, features_graph_all, \
                                features_text2d_all, feature_conf_all, feature_text3d_all, return_sim=False, w_gt=0.25, w_ct=0.25):
        
    

        bs = features_graph.size(0)
        sim_g2t = torch.mm(features_graph, features_text2d_all.transpose(0, 1))
        logits_per_graph = sim_g2t / self.temperature_gt

        sim_t2g = torch.mm(features_text2d, features_graph_all.transpose(0, 1))
        logits_per_text2d = sim_t2g / self.temperature_gt

        sim_c2t = torch.mm(feature_conf, feature_text3d_all.transpose(0, 1))
        logits_per_conf = sim_c2t / self.temperature_ct


        sim_t2c = torch.mm(feature_text3d, feature_conf_all.transpose(0, 1))
        logits_per_text3d = sim_t2c / self.temperature_ct

        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text2d = F.cross_entropy(logits_per_text2d, labels)

        loss_conf = F.cross_entropy(logits_per_conf, labels)
        loss_text3d = F.cross_entropy(logits_per_text3d, labels)

        loss1= (loss_graph + loss_text2d) / 2
        loss2= (loss_conf + loss_text3d) / 2

        loss = w_gt*loss1 + w_ct*loss2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text2d[:, rank*bs:rank*bs+bs], \
                    logits_per_conf[:, rank*bs:rank*bs+bs], logits_per_text3d[:, rank*bs:rank*bs+bs], loss1, loss2,  loss
        else:
            return loss



        
    def forward(self, batch):
        graph_batch, text2d_tokens, text2d_mask, conf_batch, text3d_tokens, text3d_mask, graph_batch_d2d3, conf_batch_d2d3, mol_batch, pro_batch= batch 
        batch_node, batch_mask2d = self.graph_encoder(graph_batch) #返回2个
        batch_conf, batch_mask3d = self.conf_encoder(*conf_batch) # *tuple
        batch_node_d2d3, batch_mask2d_d2d3 = self.graph_encoder(graph_batch_d2d3)
        batch_conf_d2d3, batch_mask3d_d2d3 = self.conf_encoder(*conf_batch_d2d3)
        batch_conf_mol, batch_conf_mask3d_mol = self.conf_encoder(*mol_batch)
        batch_conf_pro, batch_conf_mask3d_pro = self.pro_encoder(*pro_batch)

        batch_node = self.ln_graph(batch_node)
        graph_feats = self.graph_proj(batch_node)
  
        text2d_output = self.Qformer.bert(text2d_tokens, attention_mask=text2d_mask, return_dict=True) # ？：shape = [B, n_max, D]
        text2d_feats = self.text_proj(text2d_output.last_hidden_state[:, 0, :])
 
        batch_conf = self.ln_conf(torch.mean(batch_conf, dim=1))#：shape = [B, n_max, D] [CLS] Token
        conf_feats = self.conf_proj(batch_conf)
        text3d_output = self.Qformer.bert(text3d_tokens, attention_mask=text3d_mask, return_dict=True) 
        text3d_feats = self.text_proj(text3d_output.last_hidden_state[:, 0, :])

        batch_node_d2d3 = self.ln_graph(batch_node_d2d3)
        graph_feats_d2d3 = self.graph_proj(batch_node_d2d3)

        batch_conf_d2d3 = self.ln_conf(torch.mean(batch_conf_d2d3, dim=1))###：shape = [B, n_max, D] [CLS] Token
        conf_feats_d2d3 = self.conf_proj(batch_conf_d2d3)

        batch_conf_mol = self.ln_conf(torch.mean(batch_conf_mol, dim=1))###：shape = [B, n_max, D] [CLS] Token
        conf_feats_mol = self.conf_proj(batch_conf_mol)
        batch_conf_pro = self.ln_pro(torch.mean(batch_conf_pro, dim=1))###：shape = [B, n_max, D] [CLS] Token
        conf_feats_pro = self.pro_proj(batch_conf_pro)


      
        text2d_feats, graph_feats = F.normalize(text2d_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        text3d_feats, conf_feats = F.normalize(text3d_feats, p=2, dim=-1), F.normalize(conf_feats, p=2, dim=-1)


        graph_feats_d2d3, conf_feats_d2d3 = F.normalize(graph_feats_d2d3, p=2, dim=-1), F.normalize(conf_feats_d2d3, p=2, dim=-1)
        conf_feats_mol, conf_feats_pro = F.normalize(conf_feats_mol, p=2, dim=-1), F.normalize(conf_feats_pro, p=2, dim=-1)

        text2d_feats_all, graph_feats_all = pl_concat_all_gather(text2d_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        text3d_feats_all, conf_feats_all = pl_concat_all_gather(text3d_feats), pl_concat_all_gather(conf_feats)
        
        graph_feats_d2d3_all, conf_feats_d2d3_all = pl_concat_all_gather(graph_feats_d2d3), pl_concat_all_gather(conf_feats_d2d3)
        
        conf_feats_mol_all, conf_feats_pro_all =  pl_concat_all_gather(conf_feats_mol), pl_concat_all_gather(conf_feats_pro)

        sim_g2t, sim_t2g, sim_c2t, sim_t2c, loss2D, loss3D, loss_contra1 = self.contrastive(graph_feats, text2d_feats, conf_feats, text3d_feats,\
                                            graph_feats_all, text2d_feats_all, conf_feats_all, text3d_feats_all, return_sim=True, w_gt=0.25, w_ct=0.25)
        sim_d22d3, sim_d32d2, sim_mol2pro, sim_pro2mol, loss_d2d3, loss_molpro, loss_contra2 = self.contrastive(graph_feats_d2d3, conf_feats_d2d3, conf_feats_mol, conf_feats_pro,\
                                    graph_feats_d2d3_all, conf_feats_d2d3_all, conf_feats_mol_all, conf_feats_pro_all, return_sim=True, w_gt=0.25, w_ct=0.25)
        loss_contra = loss_contra1 + loss_contra2
        return loss_contra, loss2D, loss3D, loss_d2d3, loss_molpro



