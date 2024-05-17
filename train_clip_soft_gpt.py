import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
# from compGCN import CompGraphConv
import torch.nn as nn
# import clip
import collections
import copy
from PIL import Image
# 设置输出选项，取消省略号
# torch.set_printoptions(threshold=float('inf'))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from NewModel import *
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import open_clip
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import open_clip
from open_clip import *
from typing import Any, Dict, Optional, Tuple, Union
import torch
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor

torch.manual_seed(42)
BATCH_SIZE = 128
EPOCH = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
print(torch.cuda.is_available())
# model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
# _, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

model = Pair_CLIP_SI().to(device)#.cuda()

# # print(next(model.parameters()).device)
# # 将打印的内容写入文本文件
# with open('model_structure.txt', 'w') as f:
#     f.write(str(model))

class image_title_dataset(Dataset):
    def __init__(self, data_csv, si_global_path, si_local_path, stv_path, poi_text_path):
        self.si_global_path = si_global_path
        self.si_local_path = si_local_path
        self.poi_text_path = poi_text_path
        self.stv_path = stv_path
        self.data = pd.read_csv(data_csv)
        self.img_name_list = list(self.data['stv_img_name1'])
        self.text_list = list(self.data['text'])
        # 配置微调参数
        self.gpt_model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_name,local_files_only=True)
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 创建图像特征提取器
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-384')#'vit',local_files_only=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.img_name_list[idx])
        sat_global_path = self.si_global_path + self.img_name_list[idx].split('_')[0] + '_' + self.img_name_list[idx].split('_')[1] + '.png'
        # sat_local_path = self.si_local_path + self.img_name_list[16000+idx] #text

        poi_inputs = self.poi_text_path + self.img_name_list[idx].split('_')[0] + '_' + self.img_name_list[idx].split('_')[1] + '_all_info_text.txt'
        # if not os.path.exists(poi_inputs):
        #     poi_inputs = self.tokenizer('There is no info here.')
        #     # poi_inputs = 'There is no poi here.'
        # else:
        # tmp_line = ''
        # with open(poi_inputs,'r') as f:
        #     for line in f:
        #         tmp_line += line
        # # poi_inputs = tmp_line
        # poi_inputs = self.tokenizer(tmp_line)

        # si_global_image = preprocess_train(Image.open(sat_global_path)) # Image from PIL module
        # 图像预处理
        si_global_image = self.feature_extractor(Image.open(sat_global_path), return_tensors="pt")
        # si_local_image = preprocess(Image.open(sat_local_path)) # Image from PIL module
        # stv_image = preprocess(Image.open(stv_path)) # Image from PIL module

        title = self.text_list[idx]# clip.tokenize(self.text_list[idx])
        # text = "<CLS> " + text
        # self.pad_token = self.tokenizer.eos_token
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        inputs = self.tokenizer(title,max_length=128,padding='max_length',truncation=True, return_tensors='pt')#,truncation=True ,max_length=128,padding='max_length'
        # 加载预训练的BERT模型和分词器
        # tmp = self.tokenizer.batch_encode_plus([text], add_special_tokens=True, max_length=512,padding='max_length', return_attention_mask=True)
        # title = torch.tensor(inputs['input_ids']).squeeze(0)
        # attention_mask = torch.tensor(inputs['attention_mask'])
        # print(inputs)
        # print(si_global_image)
        return inputs, si_global_image['pixel_values'].cuda().to(device)

# train_file = '../STV_BJ_40.csv'
# train_file = 'SAT_STV_concat_text_single_sat_global_abstract.csv'
#train_file = 'SAT_OSM_text_gpt_description_tmp05.csv'
train_file = 'SAT_OSM_text_gpt_description_tmp1_update_consize1.csv'
# use your own data
dataset = image_title_dataset(train_file,'../CLIP/BJ_sat_used/','../SAT_STV_file_all/sat_patch/','SAT_STV_file_sep/STV/','OSM/process_osm/concat_text/')
# dataset = image_title_dataset(train_file, ' ','../STV_response_40_BJ_16000_24000/','../STV_BJ_40/')
#STV_response_40_BJ_16000_24000
train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle=True,drop_last=True) #True  Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.1)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# add your own code to track the training progress.
cnt = 0
# Put the model into training mode.
model.train()
# init_logit_scale = np.log(1 / 0.07)
# init_logit_scale_tensor = torch.tensor(init_logit_scale).to(device)
# logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale_tensor).to(device)
for epoch in range(EPOCH):
    for batch in train_dataloader :
        print('EPOCH: ',epoch) #,'batch: ',batch
        optimizer.zero_grad()

        texts,images = batch
        # images= images.to(device)
        texts = texts.to(device)
        print(images.shape)

        # texts = texts.squeeze(1)
        # poi_texts = poi_texts.to(device)
        # poi_texts = poi_texts.squeeze(1)

        ###################################################################################
        # _,poi_text_emb,_ = model.forward(_,poi_texts)
        # poi_text_emb = model.encode_text(poi_texts,normalize=True)
        # print('img_emb: ',img_emb)
        # print('text_emb: ',text_emb)

        # del img_emb,text_emb
        # print(images.shape)
        # print(texts.shape)

        # a2a = logit_scale.exp() * poi_text_emb @ poi_text_emb.T
        # poi_text_emb = poi_text_emb / poi_text_emb.norm(dim=1, keepdim=True)
        # a2a = torch.einsum('ai, ci->ac', poi_text_emb, poi_text_emb)
        # a2a = F.softmax(a2a, dim=1)
        # a2a = model.text2text(poi_texts)
        # a2a = F.softmax(a2a, dim=1)
        # print('a2a: ',a2a.shape)
        # print(a2a)
        ###################################################################################

        # Forward pass
        logits_per_image, logits_per_text  = model.get_logits(images.squeeze(1), texts)

        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        # print(ground_truth)

        # diag_a2a = torch.eye(len(images),device=device)
        # # print('diag_a2a: ',diag_a2a)

        # a2a = diag_a2a*0.9 + a2a*0.1
        # # print('a2a: ',a2a)
        # # print(F.softmax(logits_per_text, dim=1))
        # # kl = F.kl_div(F.log_softmax(a2a, dim=1), F.softmax(logits_per_text, dim=1), reduction='batchmean')
        # kl = F.kl_div(torch.log(a2a), F.softmax(logits_per_text, dim=1), reduction='batchmean')
        
        con_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss = con_loss #kl + 
        # total_loss = con_loss
      
        total_loss.backward()
        # convert_models_to_fp32(model)
        optimizer.step()
      # convert_weights(model)

    # Update the learning rate.
    #   scheduler.step()
        
        # print(total_loss.item()) 
        print(con_loss.item(), total_loss.item())  #, loss2_1.item(), loss2_2.item() kl.item(),
    #   # 获取 image_encoder_SI 中某一层的参
    #   # 获取 image_encoder_SI 中某一层的参数
    #   image_encoder_parameters = model.image_encoder_SI.parameters()

    # # 获取 encoder_TEXT 中某一层的参数
    #   encoder_text_parameters = model.encoder_TEXT.parameters()
    #   for name, param in model.image_encoder_SI.named_parameters():
    #     if "ln_post" in name:
    #       print(name, param)
    #       break

    #     # 获取 encoder_TEXT 中某一层的参数
    #   for name, param in model.encoder_TEXT.named_parameters():
    #     if "ln_f" in name:
    #       print(name, param)
    #       break
    if epoch == 0 or epoch==10 or epoch == 30 or epoch == 50 or epoch == 99:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                # }, 'model_checkpoint_soft4/'+str(epoch)+'model_test.pt') #just change to your preferred folder/filename
            # }, 'model_checkpoint_soft_no_soft_gpt3/'+str(epoch)+'model_test.pt') #just change to your preferred folder/filename
        }, 'model_checkpoint_soft_no_soft_gpt_text_update_consize/'+str(epoch)+'model_test.pt') #just change to your preferred folder/filename
# soft3:  a2a = diag_a2a*0.7 + a2a*0.3
# soft4: a2a = diag_a2a*0.9 + a2a*0.1
