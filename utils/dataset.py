import os
import torch
import pandas as pd
from monai.transforms import (Compose, NormalizeIntensityd, RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class QaTa(Dataset):
    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):
        super(QaTa, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])
        self.root_path = root_path
        self.image_size = image_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path,'Images',self.image_list[idx])
        gt = os.path.join(self.root_path, 'GTs', 'mask_' + self.image_list[idx])
        
        caption = self.caption_list[idx]
        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']
        gt = torch.where(gt==255,1,0)

        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 

        return ([image, text], gt)

    def transform(self,image_size=[224,224]):
        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])

        return trans
    
    
class MosMed(Dataset):
    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):
        super(MosMed, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])
        self.root_path = root_path
        self.image_size = image_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        trans = self.transform(self.image_size)
        
        image = os.path.join(self.root_path, 'Images', self.image_list[idx])
        
        gt = os.path.join(self.root_path, 'GTs', self.image_list[idx])
        caption = self.caption_list[idx]
        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']

        gt = torch.where(gt==255,1,0)
        gt = (torch.sum(gt, axis=0)/3.0).int()
        gt = torch.unsqueeze(gt, 0)
    
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        
        return ([image, text], gt)

    def transform(self,image_size=[224,224]):
        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])

        return trans


class Kvasir(Dataset):
    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):
        super(Kvasir, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        self.root_path = root_path
        self.image_size = image_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        trans = self.transform(self.image_size)
        
        image = os.path.join(self.root_path, 'Images', self.image_list[idx])
        
        gt = os.path.join(self.root_path, 'GTs', self.image_list[idx])
        caption = self.caption_list[idx]
        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']
    
        
        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']

        gt = torch.where(gt==255,1,0)
        
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        
        return ([image, text], gt)

    def transform(self,image_size=[224,224]):
        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])

        return trans