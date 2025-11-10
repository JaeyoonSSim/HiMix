import torch
import utils.config as config
import random
import pytorch_lightning as pl   

from torch.optim import lr_scheduler
from engine.wrapper import *
from torch.utils.data import DataLoader
from utils.dataset import QaTa, MosMed, Kvasir 
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

import os
# os.environ["CUDA_VISBLE_DEVICES"] = "7"

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg


if __name__ == '__main__':
    args = get_parser()
    print("cuda:",torch.cuda.is_available())
    
    seed=0
    torch.manual_seed(seed) # Pytorch randomness
    np.random.seed(seed) # Numpy randomness
    random.seed(seed) # Python randomness
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Current GPU randomness
        torch.cuda.manual_seed_all(seed) # Multi GPU randomness

    ## 0. setting dataloader
    if args.data == 'qata':
        ds_train = QaTa(csv_path=args.train_csv_path,
                        root_path=args.train_root_path,
                        tokenizer=args.bert_type,
                        image_size=args.image_size,
                        mode='train')
        ds_valid = QaTa(csv_path=args.valid_csv_path,
                        root_path=args.valid_root_path,
                        tokenizer=args.bert_type,
                        image_size=args.image_size,
                        mode='val')
    elif args.data == 'mosmed':
        ds_train = MosMed(csv_path=args.train_csv_path,
                        root_path=args.train_root_path,
                        tokenizer=args.bert_type,
                        image_size=args.image_size,
                        mode='train')

        ds_valid = MosMed(csv_path=args.valid_csv_path,
                        root_path=args.valid_root_path,
                        tokenizer=args.bert_type,
                        image_size=args.image_size,
                        mode='val')
    elif args.data == 'kvasir':
        ds_train = Kvasir(csv_path=args.train_csv_path,
                        root_path=args.train_root_path,
                        tokenizer=args.bert_type,
                        image_size=args.image_size,
                        mode='train')

        ds_valid = Kvasir(csv_path=args.valid_csv_path,
                        root_path=args.valid_root_path,
                        tokenizer=args.bert_type,
                        image_size=args.image_size,
                        mode='val')

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)
    
    model = HiMixWrapper(args)

    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
                dirpath=args.model_save_path,
                filename=args.model_save_filename,
                monitor='val_loss',
                save_top_k=1,
                mode='min',
                verbose=True,
    )

    early_stopping = EarlyStopping(monitor='val_loss',
                patience=args.patience,
                mode = 'min'
    )

    ## 2. setting trainer
    trainer = pl.Trainer(logger=True,
                min_epochs=args.min_epochs,
                max_epochs=args.max_epochs,
                accelerator='gpu', 
                devices=[args.de],
                callbacks=[model_ckpt, early_stopping],
                enable_progress_bar=False,
    ) 

    ## 3. start training
    print('start training')
    trainer.fit(model, dl_train, dl_valid)
    print('done training')