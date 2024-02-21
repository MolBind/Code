import torch
from torch.utils.data import Dataset
import os
import random
from loader.unimol_dataset import D3Dataset, D3Dataset_Pro


class MolClipDataset(Dataset):
    def __init__(self, root_gt, root_ct, root_d2d3, root_molpro, text_max_len, unimol_dict=None, unimol_dict_pro=None, max_atoms=256, prompt='', return_prompt=False):
        super(MolClipDataset, self).__init__()
        self.prompt = prompt
        self.return_prompt = return_prompt

        self.root_gt = root_gt
        self.root_ct = root_ct
        self.root_d2d3 = root_d2d3
        self.root_molpro = root_molpro
        self.text_max_len = text_max_len

        self.graph_name_list_2d = os.listdir(root_gt+'graph/')
        self.graph_name_list_2d.sort()
        self.text_name_list_2d = os.listdir(root_gt+'text/')
        self.text_name_list_2d.sort()


        self.text_name_list_3d = os.listdir(root_ct+'text/')
        self.text_name_list_3d=sorted(self.text_name_list_3d, key=lambda x: int(x.split('.')[0].split('_')[1]))
        self.tokenizer = None
        target_path = os.path.join(root_ct, 'unimol_mol.lmdb')
        self.d3_dataset = D3Dataset(target_path, unimol_dict, max_atoms)
        assert len(self.d3_dataset) == len(self.text_name_list_3d),print(len(self.d3_dataset),len(self.text_name_list_3d))

        self.graph_name_list_d2d3 = os.listdir(root_d2d3+'graph/')
        self.graph_name_list_d2d3 = sorted(self.graph_name_list_d2d3, key=lambda x: int(x.split('.')[0].split('_')[1]))
        d3_d2d3_path = os.path.join(root_d2d3, 'unimol_mol.lmdb')
        self.d3_dataset_d2d3 = D3Dataset(d3_d2d3_path, unimol_dict, max_atoms)

        self.d3_dataset_molpro = D3Dataset(os.path.join(root_molpro, 'ligand.lmdb'), unimol_dict, max_atoms)
        self.pro_dataset_molpro = D3Dataset_Pro(os.path.join(root_molpro, 'pocket.lmdb'), unimol_dict_pro, max_atoms)
        assert len(self.d3_dataset_molpro) == len(self.pro_dataset_molpro)
        '''
        with open(os.path.join(root, 'error_ids.txt'), 'r') as f:
            error_ids = set(json.loader(f))
        self.graph_name_list_2d = [data for i, data in enumerate(self.graph_name_list_2d) if i not in error_ids]
        self.smiles_name_list_2d = [data for i, data in enumerate(self.smiles_name_list_2d) if i not in error_ids]
        self.text_name_list_2d = [data for i, data in enumerate(self.text_name_list_2d) if i not in error_ids]
        '''
        self.permutation = None
    
    def shuffle(self):
        self.permutation = torch.randperm(len(self)).numpy()
        return self

    def __len__(self):
        return max(len(self.text_name_list_2d), len(self.d3_dataset_molpro), len(self.graph_name_list_d2d3))

    def get_2d(self, index):
        if index >= len(self.text_name_list_2d):
            index = index % len(self.text_name_list_2d)
        graph_name, text_name = self.graph_name_list_2d[index], self.text_name_list_2d[index]
        graph_path = os.path.join(self.root_gt, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        text_path = os.path.join(self.root_gt, 'text', text_name)

        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
            text = ' '.join(lines) + '\n'
            text, mask = self.tokenizer_text(text)
        
        assert not self.return_prompt
        return data_graph, text.squeeze(0), mask.squeeze(0)
    

    def get_3d(self, index):
        if index >= len(self.text_name_list_3d):
            index = index % len(self.text_name_list_3d)
        atom_vec, coordinates, edge_type, dist, smiles = self.d3_dataset[index]
        text_path = os.path.join(self.root_ct, 'text', self.text_name_list_3d[index])
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
        text = ' '.join(lines) + '\n'
        text, mask = self.tokenizer_text(text)
        return (atom_vec, coordinates, edge_type, dist, smiles), text.squeeze(0), mask.squeeze(0)
    def get_d2d3(self, index):
        if index >= len(self.graph_name_list_d2d3):
            index = index % len(self.graph_name_list_d2d3)
        graph_name = self.graph_name_list_d2d3[index]
        # loader and process graph
        graph_path = os.path.join(self.root_d2d3, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        atom_vec, coordinates, edge_type, dist, smiles = self.d3_dataset_d2d3[index]

        return data_graph, (atom_vec, coordinates, edge_type, dist, smiles)



    def get_molpro(self, index):
        if index >= len(self.d3_dataset_molpro):
            index = index % len(self.d3_dataset_molpro)
        atom_vec_mol, coordinates_mol, edge_type_mol, dist_mol, smiles = self.d3_dataset_molpro[index]
        atom_vec_pro, coordinates_pro, edge_type_pro, dist_pro, residues= self.pro_dataset_molpro[index]
        return (atom_vec_mol, coordinates_mol, edge_type_mol, dist_mol, smiles), (atom_vec_pro, coordinates_pro, edge_type_pro, dist_pro, residues)

    def __getitem__(self, index):
        if self.permutation is not None:
            index = self.permutation[index]
        pair_gt = self.get_2d(index)
        pair_ct = self.get_3d(index)
        pair_d2d3 = self.get_d2d3(index)
        pair_molpro = self.get_molpro(index)
        return pair_gt, pair_ct, pair_d2d3, pair_molpro

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask

