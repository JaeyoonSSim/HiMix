import argparse
import torch
import pytorch_lightning as pl  
import utils.config as config

from torch.utils.data import DataLoader
from engine.wrapper import *
from utils.dataset import QaTa, MosMed, Kvasir

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
    dataset_name = args.data
    # load model
    model = HiMixWrapper(args)

    checkpoint = torch.load(f'./save_model/{dataset_name}/{args.model_save_filename}.ckpt',\
                    map_location='cpu')["state_dict"]
    # del checkpoint["model.text_encoder.model.bert.embeddings.position_ids"]
    model.load_state_dict(checkpoint, strict=False)
    
    # dataloader
    if args.data == 'qata':
        ds_test = QaTa(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')
    elif args.data == 'mosmed':
        ds_test = MosMed(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')
    elif args.data == 'kvasir':
        ds_test = Kvasir(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')
    
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=True, num_workers=8)
    
    trainer = pl.Trainer(accelerator='gpu',devices=[args.de])
    model.eval()
    trainer.test(model, dl_test)