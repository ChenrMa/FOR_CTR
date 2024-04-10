import os
import torchaudio
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import torchvision.models as models
import math
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
import pandas as pd
import torch.fft as fft


class Video_Audio_encoder(torch.nn.Module):
    def __init__(self, args, video_dict):
        super(Video_Audio_encoder, self).__init__()
        self.video = "video"
        self.audio = "audio"
        self.args = args
        self.video_dict = video_dict

        # video
        self.video_dir = os.path.join(self.video, args.dataset)
        self.video_linear = torch.nn.Linear(1000, 128).to(self.args.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.video_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(
            self.args.device)
        self.video_emb = torch.zeros(len(video_dict), 140, 1000).to(self.args.device)

        # audio
        self.audio_linear = torch.nn.Linear(128, 128).to(self.args.device)
        self.audio_dir = os.path.join(self.audio, args.dataset)
        self.audio_model = torch.hub.load('vggish', 'vggish', source='local', device=self.args.device)
        self.audio_emb = torch.zeros(len(video_dict), 58, 128).to(self.args.device)

        # other
        self.last_layernorm = torch.nn.LayerNorm(args.emb_dim,elementwise_affine=False)
        self.fc_attention_audio = torch.nn.Linear(128, 1).to(self.args.device)
        self.fc_attention_video = torch.nn.Linear(128, 1).to(self.args.device)
        self.pop = torch.nn.Parameter(torch.rand(128, requires_grad=True))
        self.unpop = torch.nn.Parameter(torch.rand(128, requires_grad=True))
        self.pop_linear_high = torch.nn.Linear(128,128).to(self.args.device)
        self.unpop_linear_high = torch.nn.Linear(128,128).to(self.args.device)
        self.pop_linear_low = torch.nn.Linear(128,128).to(self.args.device)
        self.unpop_linear_low = torch.nn.Linear(128,128).to(self.args.device)
        self.attention_sigmoid = torch.nn.Sigmoid().to(self.args.device)
        self.dropout = torch.nn.Dropout(p=0.5).to(self.args.device)
        self.fusion = torch.nn.Linear(256, 8).to(self.args.device)

        # classify
        num_i = 128
        num_h = 32
        num_o = 1
        self.linear1 = torch.nn.Linear(num_i, num_h).to(self.args.device)
        self.relu = torch.nn.ReLU().to(self.args.device)
        self.linear2 = torch.nn.Linear(num_h, num_h).to(self.args.device)  
        self.relu2 = torch.nn.ReLU().to(self.args.device)
        self.linear3 = torch.nn.Linear(num_h, num_o).to(self.args.device)
        self.classify_sigmoid = torch.nn.Sigmoid().to(self.args.device)
        self.W = nn.Parameter(torch.rand(1, requires_grad=True)).to(self.args.device)

        with torch.no_grad():
            self.video_pre()
            self.audio_pre()

    def forward(self, train):
        train_pos = [self.video_dict[i] for i in train]
        video_emb_ori = self.relu(self.video_linear(self.video_emb[train_pos]))
        video_emb = video_emb_ori.view(video_emb_ori.size(0), -1)
        audio_emb_ori = self.relu(self.audio_linear(self.audio_emb[train_pos]))
        audio_emb = audio_emb_ori.view(audio_emb_ori.size(0), -1)
        #SVD
        #video
        u, s, v = torch.linalg.svd(video_emb, full_matrices=False)
        half = int(len(s) * self.args.video_s)
        high_s = torch.zeros(s.shape).to(self.args.device)
        high_s[:half] = s[:half]
        low_s = torch.zeros(s.shape).to(self.args.device)
        low_s[half:] = s[half:]
        video_high_matrix = (u @ torch.diag_embed(high_s) @ v ).view_as(video_emb_ori)
        video_low_matrix = (u @ torch.diag_embed(low_s) @ v ).view_as(video_emb_ori)
        #audio
        u, s, v = torch.linalg.svd(audio_emb, full_matrices=False)
        half = int(len(s) * self.args.audio_s)
        high_s = torch.zeros(s.shape).to(self.args.device)
        high_s[:half] = s[:half]
        low_s = torch.zeros(s.shape).to(self.args.device)
        low_s[half:] = s[half:]
        audio_high_matrix = (u @ torch.diag_embed(high_s) @ v ).view_as(audio_emb_ori)
        audio_low_matrix = (u @ torch.diag_embed(low_s) @ v ).view_as(audio_emb_ori)

        #attention
        self.video_high_emb = self.video_attention(video_high_matrix)
        self.video_low_emb = self.video_attention(video_low_matrix)
        self.audio_high_emb = self.audio_attention(audio_high_matrix)
        self.audio_low_emb = self.audio_attention(audio_low_matrix)
        
        #pop and unpop
        s_vh = self.PopAndUnpop_high(self.video_high_emb)
        s_vl = self.PopAndUnpop_low(self.video_low_emb)
        score = self.W * s_vh + (1 - self.W) * s_vl
        #audio score
        s_ah = self.PopAndUnpop_high(self.audio_high_emb)
        s_al = self.PopAndUnpop_low(self.audio_low_emb)
        #fusion
        s_ch = torch.diag(torch.matmul(self.video_high_emb,self.audio_high_emb.T)).reshape(score.shape)
        s_cl = torch.diag(torch.matmul(self.video_low_emb, self.audio_low_emb.T)).reshape(score.shape)
        cos = s_cl + s_ch
        cos = self.attention_sigmoid(cos)
        output = s_ah + score + cos
        score_dict = dict(zip(train,output))
        return score_dict

    def video_attention(self,embedding):
        drop = self.dropout(embedding) #304*140*128
        weight = self.attention_sigmoid(self.fc_attention_video(drop)) #304*140*1
        weight = F.softmax(weight, dim=1) #304*140*1
        attention_emb = torch.bmm(drop.transpose(1, 2), weight).squeeze(2)
        attention_emb = self.last_layernorm(attention_emb)
        return attention_emb

    def audio_attention(self,embedding):
        drop = self.dropout(embedding) #304*140*128
        weight = self.attention_sigmoid(self.fc_attention_audio(drop)) #304*140*1
        weight = F.softmax(weight, dim=1) #304*140*1
        attention_emb = torch.bmm(drop.transpose(1, 2), weight).squeeze(2)
        attention_emb = self.last_layernorm(attention_emb)
        return attention_emb

    def PopAndUnpop_high(self,embedding):
        pop = self.relu(self.pop_linear_high(embedding))
        unpop = self.relu(self.unpop_linear_high(embedding))
        s_p = self.mlp(pop)
        s_u = self.mlp(unpop)
        return s_p - s_u

    def PopAndUnpop_low(self,embedding):
        pop = self.pop_linear_low(embedding)
        unpop = self.unpop_linear_low(embedding)
        s_p = self.mlp(pop)
        s_u = self.mlp(unpop)
        return s_p - s_u

    def mlp(self,embedding):
        hidden1 = self.linear1(embedding)
        hidden1 = self.relu(hidden1)
        hidden2 = self.linear2(hidden1)
        hidden2 = self.relu(hidden2)
        output = self.linear3(hidden2)
        return output

    def MaxAndMin(self,train):
        data = pd.read_excel('dataset/video.xlsx', index_col='col_name')
        ctrs = {}
        for i in train:
            CTR = data.loc[i, self.args.dataset]
            ctrs[i] = CTR
        a = sorted(ctrs.items(), key=lambda x: x[1])
        nums = self.args.num
        self.max_video = [i[0] for i in a[-nums:]]
        self.min_video = [i[0] for i in a[:nums]]

    def con_loss(self,train):
        max_index = [train.index(i) for i in self.max_video]
        min_index = [train.index(i) for i in self.min_video]
        audio_pos = self.audio_high_emb[max_index]
        audio_neg = self.audio_high_emb[min_index]
        audio_low_pos = self.audio_low_emb[max_index]
        audio_low_neg = self.audio_low_emb[min_index]
        s_ap = self.PopAndUnpop_high(audio_pos)
        s_an = self.PopAndUnpop_high(audio_neg)
        s_alp = self.PopAndUnpop_high(audio_low_pos)
        s_aln = self.PopAndUnpop_high(audio_low_neg)
        loss_audio = torch.mean(-torch.log(1e-8 + torch.sigmoid(s_ap))-torch.log(1e-8 + (1 - torch.sigmoid(s_an))))
        loss_audio_low = torch.mean(-torch.log(1e-8 + torch.sigmoid(s_alp))-torch.log(1e-8 + (1 - torch.sigmoid(s_aln))))
        video_high_pos = self.video_high_emb[max_index]
        video_high_neg = self.video_high_emb[min_index]
        video_low_pos = self.video_low_emb[max_index]
        video_low_neg = self.video_low_emb[min_index]
        s_vhp = self.PopAndUnpop_high(video_high_pos)
        s_vhn = self.PopAndUnpop_high(video_high_neg)
        s_vlp = self.PopAndUnpop_low(video_low_pos)
        s_vlp = self.PopAndUnpop_low(video_low_neg)
        loss_vh = torch.max(-torch.log(1e-8 + torch.sigmoid(s_vhp))-torch.log(1e-8 + (1 - torch.sigmoid(s_vhn))))
        loss_vl = torch.max(-torch.log(1e-8 + torch.sigmoid(s_vlp))-torch.log(1e-8 + (1 - torch.sigmoid(s_vlp))))
        audio_high_pos = self.audio_high_emb[max_index]
        audio_high_neg = self.audio_high_emb[min_index]
        audio_low_pos = self.audio_low_emb[max_index]
        audio_low_neg = self.audio_low_emb[min_index]
        s_chp = torch.diag(torch.matmul(video_high_pos,audio_high_pos.T)).reshape(s_vhp.shape)
        s_chn = torch.diag(torch.matmul(video_high_neg,audio_high_neg.T)).reshape(s_vhp.shape)
        s_clp = torch.diag(torch.matmul(video_low_pos,audio_low_pos.T)).reshape(s_vhp.shape)
        s_cln = torch.diag(torch.matmul(video_low_neg,audio_low_neg.T)).reshape(s_vhp.shape)
        loss_ch = torch.max(-torch.log(1e-8 + torch.sigmoid(s_chp))-torch.log(1e-8 + (1 - torch.sigmoid(s_chn))))
        loss_cl = torch.max(-torch.log(1e-8 + torch.sigmoid(s_clp))-torch.log(1e-8 + (1 - torch.sigmoid(s_cln))))
        loss = loss_vh + loss_vl + loss_audio + loss_ch + loss_cl + loss_audio_low
        return loss


    def video_pre(self, ):
        files = os.listdir(self.video_dir)
        for item in tqdm(files):
            frame_dir = self.video_dir + '/' + item
            frames = os.listdir(frame_dir)
            frames.sort(key=lambda x: -int(x.split(".")[0][5:]))
            nums = 1
            for frame in frames:
                if nums<180:
                    frame_path = frame_dir + '/' + frame
                    input_image = Image.open(frame_path)
                    input_tensor = self.preprocess(input_image)
                    input_batch = input_tensor.unsqueeze(0).to(
                        self.args.device)  # create a mini-batch as expected by the model
                    output = self.video_model(input_batch)
                    self.video_emb[self.video_dict[item], -nums] = output
                    nums += 1
        self.video_emb = torch.nn.functional.normalize(self.video_emb, p=2.0, dim=1, eps=1e-12, out=None)

    def audio_pre(self, ):
        files = os.listdir(self.audio_dir)
        for item in tqdm(files):
            audio_path = os.path.join(self.audio_dir, item)
            output = self.audio_model.forward(audio_path)
            self.audio_emb[self.video_dict[item[:-4]], -output.shape[0]:] = output
        self.audio_emb = torch.nn.functional.normalize(self.audio_emb, p=2.0, dim=2, eps=1e-12, out=None)