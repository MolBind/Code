
from pytorch_lightning import LightningDataModule
from loader.pretrain_dataset import MolClipDataset
from torch.utils.data import Dataset, DataLoader
from loader.unimol_dataset import D3Collater, D3Collater_Pro
from torch_geometric.loader.dataloader import Collater
from unicore.data import Dictionary
from loader.test_dataset import  RetrievalDataset_3DText, RetrievalDataset_2DText, RetrievalDataset_MolPro, RetrievalDataset_D2D3


class MyCollater:
    def __init__(self, tokenizer, text_max_len, pad_idx):
        self.pad_idx = pad_idx
        self.d3_collater = D3Collater(pad_idx)
        self.pro_collater = D3Collater_Pro(pad_idx)
        self.d2_collater = Collater([], [])
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def __call__(self, batch):

        pair_gt_list = [pair[0] for pair in batch]
        pair_ct_list = [pair[1] for pair in batch]
        pair_d2d3_list = [pair[2] for pair in batch]
        pair_molpro_list = [pair[3] for pair in batch] 
        graph_batch_raw, text2d_batch_raw, text_2d_mask = zip(*pair_gt_list)
        conf_batch_raw, text3d_batch_raw, text_3d_mask= zip(*pair_ct_list)
        graph_batch_raw_d2d3, conf_batch_raw_d2d3 = zip(*pair_d2d3_list)
        mol_batch_raw, pro_batch_raw = zip(*pair_molpro_list)


        graph_batch = self.d2_collater(graph_batch_raw)
        text2d_tokens = self.d2_collater(text2d_batch_raw)
        text2d_mask = self.d2_collater(text_2d_mask)
        text3d_tokens = self.d2_collater(text3d_batch_raw)
        text3d_mask = self.d2_collater(text_3d_mask)
        padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(conf_batch_raw)

        graph_batch_d2d3 = self.d2_collater(graph_batch_raw_d2d3)
        padded_atom_vec_d2d3, padded_coordinates_d2d3, padded_edge_type_d2d3, padded_dist_d2d3, smiles_d2d3 = self.d3_collater(conf_batch_raw_d2d3)

        padded_atom_vec_mol, padded_coordinates_mol, padded_edge_type_mol, padded_dist_mol, smiles_mol = self.d3_collater(mol_batch_raw)
        padded_atom_vec_pro, padded_coordinates_pro, padded_edge_type_pro, padded_dist_pro, residues = self.pro_collater(pro_batch_raw)

        return graph_batch, text2d_tokens, text2d_mask, (padded_atom_vec, padded_dist, padded_edge_type), text3d_tokens, text3d_mask, graph_batch_d2d3, (padded_atom_vec_d2d3, padded_dist_d2d3, padded_edge_type_d2d3),\
                            (padded_atom_vec_mol, padded_dist_mol, padded_edge_type_mol), (padded_atom_vec_pro, padded_dist_pro, padded_edge_type_pro)


class DatesetMolBind(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root2d: str = './MolBind4M/2D-Text',
        root3d: str = './MolBind4M/3D-Text',
        root_d2d3: str = './MolBind4M/3D-2D',
        root_molpro: str = './MolBind4M/3D-Protein',
        text_max_len: int = 128,
        dictionary = None,
        dictionary_pro = None,
        tokenizer=None,
        args=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.dictionary = dictionary
        self.dictionary_pro = dictionary_pro
        self.tokenizer = tokenizer
        self.args = args
    
        self.train_dataset = MolClipDataset(root2d+'/pretrain/', root3d+'/pretrain/', root_d2d3+'/pretrain/', root_molpro+'/pretrain/', text_max_len, dictionary, dictionary_pro, args.unimol_max_atoms)
        self.val_dataset = MolClipDataset(root2d + '/valid/', root3d+'/valid/', root_d2d3+'/valid/', root_molpro+'/valid/', text_max_len, dictionary, dictionary_pro, args.unimol_max_atoms) #Text-2D Graph Text-3D Graph
        self.test_dataset = MolClipDataset(root2d + '/test/', root3d+'/test/', root_d2d3+'/test/', root_molpro+'/test/', text_max_len, dictionary, dictionary_pro, args.unimol_max_atoms) #Text-2D Graph Text-3D Graph
        self.val_dataset_match_2dtext = RetrievalDataset_2DText(root2d + '/valid/', text_max_len, tokenizer, args).shuffle() #Text-2D Graph
        self.test_dataset_match_2dtext = RetrievalDataset_2DText(root2d + '/test/', text_max_len, tokenizer, args).shuffle() #Text-2D Graph

        self.val_dataset_match_3dtext = RetrievalDataset_3DText(root3d + '/valid/', text_max_len, dictionary, args.unimol_max_atoms, tokenizer, args).shuffle()
        self.test_dataset_match_3dtext = RetrievalDataset_3DText(root3d + '/test/', text_max_len, dictionary, args.unimol_max_atoms, tokenizer, args).shuffle()

        self.val_dataset_match_d2d3 = RetrievalDataset_D2D3(root_d2d3 + '/valid/', dictionary, args.unimol_max_atoms, args).shuffle()  
        self.test_dataset_match_d2d3 = RetrievalDataset_D2D3(root_d2d3 + '/test/', dictionary, args.unimol_max_atoms, args).shuffle()
        
        self.val_dataset_match_molpro = RetrievalDataset_MolPro(root_molpro + '/valid/', dictionary, dictionary_pro, args.unimol_max_atoms, args).shuffle() 
        self.test_dataset_match_molpro = RetrievalDataset_MolPro(root_molpro + '/test/', dictionary, dictionary_pro, args.unimol_max_atoms, args).shuffle()

        self.val_match_loader_2dtext = DataLoader(self.val_dataset_match_2dtext, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_2dtext.collater)

        self.test_match_loader_2dtext = DataLoader(self.test_dataset_match_2dtext, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_2dtext.collater)

        self.val_match_loader_3dtext = DataLoader(self.val_dataset_match_3dtext, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_3dtext.collater)
        self.test_match_loader_3dtext = DataLoader(self.test_dataset_match_3dtext, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_3dtext.collater)
        self.val_match_loader_d2d3 = DataLoader(self.val_dataset_match_d2d3, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_d2d3.collater)
        self.test_match_loader_d2d3 = DataLoader(self.test_dataset_match_d2d3, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_d2d3.collater)
        self.val_match_loader_molpro = DataLoader(self.val_dataset_match_molpro, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_molpro.collater)
        self.test_match_loader_molpro = DataLoader(self.test_dataset_match_molpro, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_molpro.collater)

    
    def load_unimol_dict(self):
        dictionary = Dictionary.load('./MolBind4M/unimol_dict_mol.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        return dictionary
    def load_unimol_pro_dict(self):
        dictionary = Dictionary.load('./MolBind4M/unimol_dict_pro.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        return dictionary

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.dictionary.pad())
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.dictionary.pad())
        )

        return loader


    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=96)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--root2d', type=str, default='./MolBind4M/2D-Text')
        parser.add_argument('--root3d', type=str, default='./MolBind4M/3D-Text')
        parser.add_argument('--root_d2d3', type=str, default='./MolBind4M/3D-2D')
        parser.add_argument('--root_molpro', type=str, default= './MolBind4M/3D-Protein')
        parser.add_argument('--text_max_len', type=int, default=256)
        parser.add_argument('--use_phy_eval', action='store_true', default=False)
        return parent_parser
    