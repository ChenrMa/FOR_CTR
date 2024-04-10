from utils import *
from model import *
import argparse
import numpy as np
import torch
import time
import tqdm

#

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def main():
    set_seed(2022)

    def str2bool(s):
        if s not in {'false','true'}:
            raise ValueError('not a valid boolean string')
        return s == 'true'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TT') #TT,FB,Snap,GA
    parser.add_argument('--audio_dir', default='audio')
    parser.add_argument('--video_dir', default='frame')
    parser.add_argument('--epoch', default=61)
    parser.add_argument('--emb_dim', default=128)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--lamda', default=0.02)
    parser.add_argument('--num', default=35)
    parser.add_argument('--audio_s', default=0.6)
    parser.add_argument('--video_s', default=0.5)
    parser.add_argument('--device', default='cuda:2')

    args = parser.parse_args()

    # score = load_data(args.dataset)
    # nums, train, valid, test = split_train_test(score)
    train_score, valid_score, test_score = load_data(args.dataset)
    video_dict, train, valid, test = build_video_dict(args.dataset)


    f = open(os.path.join(args.dataset, '+loss_audio_low: ' + time.strftime('%m_%d_%Hh%Mm%Ss') + '_log.txt'),'w')

    model = Video_Audio_encoder(args,video_dict).to(args.device)
    torch.save(model, args.dataset + '.pth')

    for name,param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass
    model.train()
    model.MaxAndMin(train)
    epoch_start_idx = 1

    adam_optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,betas=(0.9,0.89))
    hit_max = 0
    for epoch in range(epoch_start_idx,args.epoch):
        score_final = model(train)
        adam_optimizer.zero_grad()
        loss = va_loss(score_final, train_score)
        con_loss = model.con_loss(train)
        loss = loss + con_loss.reshape(loss.shape) * args.lamda
        loss.requires_grad_(True)
        print('epoch ' + str(epoch) + 'train loss:' + str(loss))
        loss.backward()
        adam_optimizer.step()
        torch.cuda.empty_cache()

        if epoch % 5 == 0:
            model.eval()
            score_final = model(valid)
            hit = evaluate(score_final, valid_score)
            print('epoch' + str(epoch) + 'valid hit rate: ' + str(hit))
            f.write("epoch is :" + str(epoch) + "Hit is :" + str(hit) + '\n')
            f.flush()
            if hit > hit_max:
                hit_max = hit
                torch.save(model, args.dataset + '.pth')
            model.train()
    model.eval()
    model = torch.load(args.dataset + '.pth')
    score_final = model(test)
    hit = evaluate(score_final, test_score)
    SC = Spearmans_Correlation(test,score_final,args.dataset)
    NDCG = val_NDCG(test,score_final,args.dataset)
    print("test hit: " + str(hit))
    print("max_hit: " + str(hit_max))
    print('SC: '+ str(SC))
    print('NDCG: ' + str(NDCG))
    f.write("test hit:" + str(hit)[:7] + '\n')
    f.write("max eval hit:" + str(hit_max)[:7] + '\n')
    f.write("SC:" + str(SC) + '\n')
    f.write("NDCG: " + str(NDCG) + '\n')
    f.write("test hit:" + str(hit) + '\n')
    f.write("max eval hit:" + str(hit_max))
    f.close()

if __name__ == '__main__':
    main()