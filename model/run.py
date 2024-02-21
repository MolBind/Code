import sys
sys.path.append('/root/MolBind-final/MolBind/model')
import torch
import contextlib
from contrastive import Model
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from tqdm import tqdm
from typing import Any, Dict


def precision2dtype(precision):
    if precision == '16':
        return torch.float16
    elif precision == '32':
        return torch.float32
    elif precision.find('bf16') >= 0:
        return torch.bfloat16
    else:
        raise NotImplementedError()
    

class MolbindTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        '''
        if isinstance(args, dict):
            args = AttrDict(**args)
        '''
        self.args = args
        self.rerank_cand_num = args.rerank_cand_num
        self.model = Model(args.gtm, args.lm, args.bert_name, args.temperature, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.projection_dim, args)
    
        self.save_hyperparameters(args)
        
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = batch[1].size(0)
        loss_contra, loss2D, loss3D, loss_d2d3, loss_molpro = self.model(batch)
        self.log("val_loss", float(loss_contra), batch_size=batch_size, sync_dist=True)
        self.log("val_2Dloss", float(loss2D), batch_size=batch_size, sync_dist=True)
        self.log("val_3Dloss", float(loss3D), batch_size=batch_size, sync_dist=True)
        self.log("val_d2d3loss", float(loss_d2d3), batch_size=batch_size, sync_dist=True)
        self.log("val_MolProloss", float(loss_molpro), batch_size=batch_size, sync_dist=True)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        if self.current_epoch == 100 or (self.current_epoch + 1) % self.args.retrieval_eval_epoch != 0:
            return
        if self.trainer.global_rank == 0:
            with self.maybe_autocast(precision2dtype(str(self.args.precision))):

                ### 2D-Text Retrival
                g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total = eval_retrieval_inbatch(self.model, self.val_match_loader_2dtext, self.device, mode='2dtext') ###
                self.log("val_inbatch_g2t_acc", g2t_acc, sync_dist=False)
                self.log("val_inbatch_t2g_acc", t2g_acc, sync_dist=False)
                self.log("val_inbatch_g2t_rec20", g2t_rec20, sync_dist=False)
                self.log("val_inbatch_t2g_rec20", t2g_rec20, sync_dist=False)
                #print('2D',g2t_acc, t2g_acc, g2t_rec20, t2g_rec20)

                g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device) ###
                self.log("val_fullset_g2t_acc", g2t_acc, sync_dist=False)
                self.log("val_fullset_t2g_acc", t2g_acc, sync_dist=False)
                self.log("val_fullset_g2t_rec20", g2t_rec20, sync_dist=False)
                self.log("val_fullset_t2g_rec20", t2g_rec20, sync_dist=False)

                g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total = eval_retrieval_inbatch(self.model, self.test_match_loader_2dtext, self.device, mode='2dtext') ###
                self.log("test_inbatch_g2t_acc", g2t_acc, sync_dist=False)
                self.log("test_inbatch_t2g_acc", t2g_acc, sync_dist=False)
                self.log("test_inbatch_g2t_rec20", g2t_rec20, sync_dist=False)
                self.log("test_inbatch_t2g_rec20", t2g_rec20, sync_dist=False)
                #print('2D',g2t_acc, t2g_acc, g2t_rec20, t2g_rec20)

                g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device) ###
                self.log("test_fullset_g2t_acc", g2t_acc, sync_dist=False)
                self.log("test_fullset_t2g_acc", t2g_acc, sync_dist=False)
                self.log("test_fullset_g2t_rec20", g2t_rec20, sync_dist=False)
                self.log("test_fullset_t2g_rec20", t2g_rec20, sync_dist=False)



                ### 3D-Text Retrival
                g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total = eval_retrieval_inbatch(self.model, self.val_match_loader_3dtext, self.device, mode='3dtext') ###
                self.log("val_inbatch_c2t_acc", g2t_acc, sync_dist=False)
                self.log("val_inbatch_t2c_acc", t2g_acc, sync_dist=False)
                self.log("val_inbatch_c2t_rec20", g2t_rec20, sync_dist=False)
                self.log("val_inbatch_t2c_rec20", t2g_rec20, sync_dist=False)

                g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device) ###
                self.log("val_fullset_c2t_acc", g2t_acc, sync_dist=False)
                self.log("val_fullset_t2c_acc", t2g_acc, sync_dist=False)
                self.log("val_fullset_c2t_rec20", g2t_rec20, sync_dist=False)
                self.log("val_fullset_t2c_rec20", t2g_rec20, sync_dist=False)

                g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total = eval_retrieval_inbatch(self.model, self.test_match_loader_3dtext, self.device, mode='3dtext') ###
                self.log("test_inbatch_c2t_acc", g2t_acc, sync_dist=False)
                self.log("test_inbatch_t2c_acc", t2g_acc, sync_dist=False)
                self.log("test_inbatch_c2t_rec20", g2t_rec20, sync_dist=False)
                self.log("test_inbatch_t2c_rec20", t2g_rec20, sync_dist=False)

                g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device) ###
                self.log("test_fullset_c2t_acc", g2t_acc, sync_dist=False)
                self.log("test_fullset_t2c_acc", t2g_acc, sync_dist=False)
                self.log("test_fullset_c2t_rec20", g2t_rec20, sync_dist=False)
                self.log("test_fullset_t2c_rec20", t2g_rec20, sync_dist=False)

                ### 2D-3D Retrival
                g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total = eval_retrieval_inbatch(self.model, self.val_match_loader_d2d3, self.device, mode='d2d3') ###
                self.log("val_inbatch_2d_2_3d_acc", g2t_acc, sync_dist=False)
                self.log("val_inbatch_3d_2_2d_acc", t2g_acc, sync_dist=False)
                self.log("val_inbatch_2d_2_3d_rec20", g2t_rec20, sync_dist=False)
                self.log("val_inbatch_3d_2_2d_rec20", t2g_rec20, sync_dist=False)
                #print('2D',g2t_acc, t2g_acc, g2t_rec20, t2g_rec20)

                g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device) ###
                self.log("val_fullset_2d_2_3d_acc", g2t_acc, sync_dist=False)
                self.log("val_fullset_3d_2_2d_acc", t2g_acc, sync_dist=False)
                self.log("val_fullset_2d_2_3d_rec20", g2t_rec20, sync_dist=False)
                self.log("val_fullset_3d_2_2d_rec20", t2g_rec20, sync_dist=False)

                g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total = eval_retrieval_inbatch(self.model, self.test_match_loader_d2d3, self.device, mode='d2d3') ###
                self.log("test_inbatch_2d_2_3d_acc", g2t_acc, sync_dist=False)
                self.log("test_inbatch_3d_2_2d_acc", t2g_acc, sync_dist=False)
                self.log("test_inbatch_2d_2_3d_rec20", g2t_rec20, sync_dist=False)
                self.log("test_inbatch_3d_2_2d_rec20", t2g_rec20, sync_dist=False)
                #print('2D',g2t_acc, t2g_acc, g2t_rec20, t2g_rec20)

                g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device) ###
                self.log("test_fullset_2d_2_3d_acc", g2t_acc, sync_dist=False)
                self.log("test_fullset_3d_2_2d_acc", t2g_acc, sync_dist=False)
                self.log("test_fullset_2d_2_3d_rec20", g2t_rec20, sync_dist=False)
                self.log("test_fullset_3d_2_2d_rec20", t2g_rec20, sync_dist=False)
                
                ### mol-pro Retrival
                g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total = eval_retrieval_inbatch(self.model, self.val_match_loader_molpro, self.device, mode='molpro') ###
                self.log("val_inbatch_mol2pro_acc", g2t_acc, sync_dist=False)
                self.log("val_inbatch_pro2mol_acc", t2g_acc, sync_dist=False)
                self.log("val_inbatch_mol2pro_rec20", g2t_rec20, sync_dist=False)
                self.log("val_inbatch_pro2mol_rec20", t2g_rec20, sync_dist=False)
                g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device) 
                self.log("val_fullset_mol2pro_acc", g2t_acc, sync_dist=False)
                self.log("val_fullset_pro2mol_acc", t2g_acc, sync_dist=False)
                self.log("val_fullset_mol2pro_rec20", g2t_rec20, sync_dist=False)
                self.log("val_fullset_pro2mol_rec20", t2g_rec20, sync_dist=False)

                g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total = eval_retrieval_inbatch(self.model, self.test_match_loader_molpro, self.device, mode='molpro') ###
                self.log("test_inbatch_mol2pro_acc", g2t_acc, sync_dist=False)
                self.log("test_inbatch_pro2mol_acc", t2g_acc, sync_dist=False)
                self.log("test_inbatch_mol2pro_rec20", g2t_rec20, sync_dist=False)
                self.log("test_inbatch_pro2mol_rec20", t2g_rec20, sync_dist=False)
                g2t_acc, g2t_rec20, t2g_acc, t2g_rec20 = eval_retrieval_fullset(graph_rep_total, text_rep_total, self.device) ###
                self.log("test_fullset_mol2pro_acc", g2t_acc, sync_dist=False)
                self.log("test_fullset_pro2mol_acc", t2g_acc, sync_dist=False)
                self.log("test_fullset_mol2pro_rec20", g2t_rec20, sync_dist=False)
                self.log("test_fullset_pro2mol_rec20", t2g_rec20, sync_dist=False)


    def training_step(self, batch, batch_idx):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[1].size(0) # Fixme?
        loss_contra, loss2D, loss3D, loss_d2d3, loss_molpro = self.model(batch)
        ###============== Overall Loss ===================###
        self.log("train_loss", float(loss_contra), batch_size=batch_size, sync_dist=True)
        self.log("train_2Dloss", float(loss2D), batch_size=batch_size, sync_dist=True)
        self.log("train_3Dloss", float(loss3D), batch_size=batch_size, sync_dist=True)
        self.log("train_d2d3loss", float(loss_d2d3), batch_size=batch_size, sync_dist=True)
        self.log("train_MolProloss", float(loss_molpro), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss_contra

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # evaluation
        parser.add_argument('--rerank_cand_num', type=int, default=128)
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)

        parser.add_argument('--filename', type=str, default="stage1_test")
        parser.add_argument('--store_path', type=str, default="./all_checkpoints")
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=False)
        parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=False)
        parser.add_argument('--mode', type=str, default='train')
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--projection_dim', type=int, default=512)
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--retrieval_eval_epoch', type=int, default=1)
        parser.add_argument('--save_every_n_epochs', type=int, default=10)
        return parent_parser

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)


def pad_and_concat(tensor_list):
    '''
    concat the first dimension and pad the second dimension
    tensor_list: [[B (diff), N_num, *], ...]
    '''
    device = tensor_list[0].device
    max_dim1 = max(t.shape[1] for t in tensor_list)
    sum_dim0 = sum(t.shape[0] for t in tensor_list)
    if len(tensor_list[0].shape) == 3:
        out = torch.zeros((sum_dim0, max_dim1, tensor_list[0].shape[-1]), device=device)
        i = 0
        for t in tensor_list:
            out[i:i+t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    elif len(tensor_list[0].shape) == 2:
        out = torch.zeros((sum_dim0, max_dim1), device=device)
        i = 0
        for t in tensor_list:
            out[i:i+t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    raise NotImplementedError()




@torch.no_grad()
def eval_retrieval_inbatch(model, dataloader, device=None, mode=None):
    assert isinstance(model, Model)
    model.eval()
    a2b_acc = 0
    b2a_acc = 0
    a2b_rec20 = 0
    b2a_rec20 = 0
    allcnt = 0
    
    a_rep_total = []  
    b_rep_total = []

    b_total = []
    b_mask_total = []
    
    for batch in tqdm(dataloader):
        if mode == '2dtext':
            graph, text, text_mask = batch
            b_total.append(text)
            b_mask_total.append(text_mask)
            graph = graph.to(device)
            text = text.to(device)
            text_mask = text_mask.to(device)

            graph_rep, _ = model.graph_encoder(graph) # shape = [B, D]
            graph_rep = model.ln_graph(graph_rep)
            graph_rep = model.graph_proj(graph_rep).detach()
            text_rep = model.Qformer.bert(text, text_mask, return_dict=True) # shape = [B, D]
            text_rep = model.text_proj(text_rep.last_hidden_state[:, 0, :]).detach()
            a_rep = graph_rep
            b_rep = text_rep
            b_mask = text_mask
        if mode == '3dtext':
            conf, text, text_mask = batch
            padded_atom_vec, padded_dist, padded_edge_type = conf
            #conf_batch = conf_batch.to(device)
            padded_atom_vec, padded_dist, padded_edge_type = padded_atom_vec.to(device), padded_dist.to(device), padded_edge_type.to(device)
            text = text.to(device)
            text_mask = text_mask.to(device)

            conf_rep, _ = model.conf_encoder(padded_atom_vec, padded_dist, padded_edge_type)
            conf_rep = model.ln_conf(torch.mean(conf_rep, dim=1)).detach()
            conf_rep = model.conf_proj(conf_rep)
            text_rep = model.Qformer.bert(text, text_mask, return_dict=True) # shape = [B, D]
            text_rep = model.text_proj(text_rep.last_hidden_state[:, 0, :]).detach()
            a_rep = conf_rep
            b_rep = text_rep
            b_mask = text_mask
            # print('3dtext_a_rep:',a_rep)
            # print('3dtext_b_rep:',b_rep)
        if mode == 'd2d3':
            graph_batch_d2, conf_batch_d3 = batch
            graph_batch_d2 = graph_batch_d2.to(device)
            padded_atom_vec, padded_dist, padded_edge_type = conf_batch_d3
            padded_atom_vec, padded_dist, padded_edge_type = padded_atom_vec.to(device), padded_dist.to(device), padded_edge_type.to(device)
            graph_rep_d2, _ = model.graph_encoder(graph_batch_d2)
            graph_rep_d2 = model.ln_graph(graph_rep_d2)
            graph_rep_d2 = model.graph_proj(graph_rep_d2).detach().cpu()
            conf_rep_d3, _ = model.conf_encoder(padded_atom_vec, padded_dist, padded_edge_type)    
            conf_rep_d3 = model.ln_conf(torch.mean(conf_rep_d3, dim=1))###：shape = [B, n_max, D] [CLS] Token
            conf_rep_d3 = model.conf_proj(conf_rep_d3).detach().cpu()

            a_rep = graph_rep_d2
            b_rep = conf_rep_d3
            #b_mask = text_mask
        if mode == 'molpro':
            mol_batch, pro_batch = batch
            padded_atom_vec_mol, padded_dist_mol, padded_edge_type_mol = mol_batch
            padded_atom_vec_pro, padded_dist_pro, padded_edge_type_pro = pro_batch
            padded_atom_vec_mol, padded_dist_mol, padded_edge_type_mol = padded_atom_vec_mol.to(device), padded_dist_mol.to(device), padded_edge_type_mol.to(device)
            padded_atom_vec_pro, padded_dist_pro, padded_edge_type_pro = padded_atom_vec_pro.to(device), padded_dist_pro.to(device), padded_edge_type_pro.to(device)
            mol_rep, _ = model.conf_encoder(padded_atom_vec_mol, padded_dist_mol, padded_edge_type_mol)
            mol_rep = model.ln_conf(torch.mean(mol_rep, dim=1))
            mol_rep = model.conf_proj(mol_rep).detach().cpu()
            pro_rep, _ = model.pro_encoder(padded_atom_vec_pro, padded_dist_pro, padded_edge_type_pro)
            pro_rep = model.ln_pro(torch.mean(pro_rep, dim=1))
            pro_rep = model.pro_proj(pro_rep).detach().cpu()

            a_rep = mol_rep
            b_rep = pro_rep




        sim_a2b = torch.mm(a_rep, b_rep.transpose(0, 1))
        
        

        B = sim_a2b.shape[0]
        sorted_ids = sim_a2b.argsort(descending=True).cpu()
        a2b_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        sorted_ids = sim_a2b.T.argsort(descending=True).cpu()
        b2a_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)


        a2b_acc += float((a2b_rank == 0).sum())
        b2a_acc += float((b2a_rank == 0).sum())
        a2b_rec20 += float((a2b_rank < 20).sum())
        b2a_rec20 += float((b2a_rank < 20).sum())
        
        allcnt += B

        a_rep_total.append(a_rep.cpu())
        b_rep_total.append(b_rep.cpu())

    a_rep_total = torch.cat(a_rep_total, dim=0)
    b_rep_total = torch.cat(b_rep_total, dim=0)

    a2b_acc = (a2b_acc/allcnt) * 100
    b2a_acc = (b2a_acc/allcnt) * 100
    a2b_rec20 = (a2b_rec20 / allcnt) * 100
    b2a_rec20 = (b2a_rec20 / allcnt) * 100


    return a2b_acc, b2a_acc, a2b_rec20, b2a_rec20, a_rep_total, b_rep_total


@torch.no_grad()
def eval_retrieval_fullset(a_rep, b_rep, device):    
    N = a_rep.shape[0]
    B = 8
    b_rep = b_rep.to(device)
    sim_a2b = []
    for i in tqdm(range(0, N, B)):
        l_a_rep = a_rep[i:i+B].to(device)
        l_sim_a2b = torch.mm(l_a_rep, b_rep.transpose(0, 1))
        sim_a2b.append(l_sim_a2b)
    sim_a2b = torch.cat(sim_a2b, dim=0).cpu() # shape = [N, N]
    
    rank_a2b = []
    for i in range(0, N, B):
        sorted_ids = torch.argsort(sim_a2b[i:i+B].to(device), descending=True)
        rank_a2b.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0], device=device).reshape(-1, 1)).int().argmax(dim=-1))
    rank_a2b = torch.cat(rank_a2b, dim=0)
    
    rank_b2a = []
    for i in range(0, N, B):
        sorted_ids = torch.argsort(sim_a2b.T[i:i+B].to(device), descending=True)
        rank_b2a.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0], device=device).reshape(-1, 1)).int().argmax(dim=-1))
    rank_b2a = torch.cat(rank_b2a, dim=0)
    
    a2b_acc = float((rank_a2b == 0).float().mean())
    a2b_rec20 = float((rank_a2b < 20).float().mean())
    b2a_acc = float((rank_b2a == 0).float().mean())
    b2a_rec20 = float((rank_b2a < 20).float().mean())
    a2b_acc = round(a2b_acc * 100, 2)
    a2b_rec20 = round(a2b_rec20 * 100, 2)
    b2a_acc = round(b2a_acc * 100, 2)
    b2a_rec20 = round(b2a_rec20 * 100, 2)
    return a2b_acc, a2b_rec20, b2a_acc, b2a_rec20#, sim_a2b