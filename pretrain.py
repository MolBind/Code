
import os
import argparse
import torch
import warnings

import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.run import MolbindTrainer
from model.unimol import SimpleUniMolModel
from loader.dataloader import DatesetMolBind
os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')

def init(data_loader,MolBind):
    data_loader.train_dataset.tokenizer = MolBind.model.tokenizer
    data_loader.val_dataset.tokenizer = MolBind.model.tokenizer
    data_loader.val_match_loader_2dtext.tokenizer = MolBind.model.tokenizer
    data_loader.test_match_loader_2dtext.tokenizer = MolBind.model.tokenizer
    data_loader.test_match_loader_3dtext.tokenizer = MolBind.model.tokenizer
    MolBind.val_match_loader_2dtext = data_loader.val_match_loader_2dtext
    MolBind.test_match_loader_2dtext = data_loader.val_match_loader_2dtext
    MolBind.val_match_loader_3dtext = data_loader.val_match_loader_3dtext
    MolBind.test_match_loader_3dtext = data_loader.test_match_loader_3dtext
    MolBind.val_match_loader_d2d3 = data_loader.val_match_loader_d2d3
    MolBind.test_match_loader_d2d3 = data_loader.test_match_loader_d2d3
    MolBind.val_match_loader_molpro = data_loader.val_match_loader_molpro
    MolBind.test_match_loader_molpro = data_loader.test_match_loader_molpro

def main(args):
    pl.seed_everything(args.seed)
    MolBind = MolbindTrainer(args)
    print('total params:', sum(p.numel() for p in MolBind.parameters()))
    print(args.match_batch_size)
    data_loader = DatesetMolBind(args.num_workers, args.batch_size, args.root2d, args.root3d, args.root_d2d3, args.root_molpro, args.text_max_len, MolBind.model.dictionary_mol, MolBind.model.dictionary_pro, MolBind.model.tokenizer, args)
    init(data_loader,MolBind)
    callbacks = []

    callbacks.append(plc.ModelCheckpoint(dirpath="checkpoints/"+args.filename+"/",
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_top_k=-1))
    find_unused_parameters = True
    if len(args.devices.split(',')) > 1:
        strategy = strategies.DDPSpawnStrategy(find_unused_parameters=find_unused_parameters)
    else:
        strategy = None
        args.devices = eval(args.devices)
        print(args.devices)
    logger = CSVLogger(save_dir=f'{args.store_path}/{args.filename}/')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         logger=logger
                                         )
    if args.mode == 'train':
        trainer.fit(MolBind, datamodule=data_loader)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 49
        trainer.validate(MolBind, datamodule=data_loader)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = MolbindTrainer.add_model_specific_args(parser)
    parser = DatesetMolBind.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)
    parser.set_defaults(accelerator='gpu',
                        devices='0,1,2,3',
                        precision='bf16',
                        max_epochs=120,
                        accumulate_grad_batches=4,
                        val_check_interval=0.1)

    args = parser.parse_args()
 
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

