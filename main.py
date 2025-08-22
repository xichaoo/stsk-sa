import os.path
import random
from torch.utils.data import Dataset
import torch
from model import *
from torchvision.transforms import functional as F
import copy
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score, roc_auc_score, confusion_matrix
from logger import create_logger
from utils import *
import argparse
from copy import deepcopy
from torcheval.metrics import MulticlassAUROC

class AlignedConcDataset_1d(Dataset):

    def __init__(self, cfg, data_dir=None,cls_num=None):
        self.cfg = cfg
        self.data=pd.read_csv(data_dir)
        self.data_dir = data_dir
        self.cls_num = cls_num
        
    def __len__(self):
        #print(len(self.data['gt']))
        #self.data_index=np.load(self.data_idx)
        return len(self.data)

    def __getitem__(self, index):
        inputx = self.data.iloc[index,1:-1].to_numpy()
        inputx=np.asarray(inputx,np.float32)
        #print(inputx.shape)
        y = self.data.iloc[index,-1]
        label = self.cls_num[y]
        #print(inputx)
        #print(y)
        inputx = torch.from_numpy(inputx)
        label = torch.tensor(label).long()
        return inputx,label
    

# def compute_matrix(ypred,tgt,name=None):
#     accs = accuracy_score(ypred.detach().cpu().numpy(),tgt.detach().cpu().numpy())
#     g_mean = recall_score(ypred.detach().cpu().numpy(),tgt.detach().cpu().numpy(),average='micro')
#     auc = roc_auc_score(ypred.detach().cpu().numpy(),tgt.detach().cpu().numpy(),average='micro')
#     f1_scores = f1_score(ypred.detach().cpu().numpy(),tgt.detach().cpu().numpy(),average='micro')
#     mastrix={
#         'accs':accs,
#         'g_mean':g_mean,
#         'auc':auc,
#         'f1_score':f1_scores
#     }
#     return mastrix

def compute_g_mean(conf_matrix):
    # Compute sensitivity (recall) and specificity for each class
    sensitivity = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + 1e-6)
    specificity = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=0) + 1e-6)
    
    # Compute G-Mean
    g_mean = np.sqrt(sensitivity * specificity).mean()
    return g_mean

    # 计算每行的softmax
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def compute_matrix(ypred, tgt, predpro, name=None):
    ypred_np = ypred.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()
    predpro_np = predpro.detach().cpu().numpy()

    # 对每一行进行softmax操作
    softmax_array = softmax(predpro_np) 

    accs = accuracy_score(tgt_np, ypred_np)
    # auc = roc_auc_score(tgt_np, ypred_np, average='micro')
    f1_scores = f1_score(tgt_np, ypred_np, average='macro')

    # Compute confusion matrix
    conf_matrix = confusion_matrix(tgt_np, ypred_np)
    
    # Compute G-Mean
    g_mean = compute_g_mean(conf_matrix)

    # Log shapes for debugging
    # print(f"tgt_np shape: {tgt_np.shape}, ypred_np shape: {softmax_array.shape}")
    # print(tgt_np)
    # print(predpro_np)
    # print(softmax_array)

    # metric = MulticlassAUROC(num_classes=7)
    # metric.update(torch.tensor(ypred_np), torch.tensor(tgt_np))
    # auc = metric.compute()

    # if len(np.unique(tgt_np)) > 1:
    #     auc = roc_auc_score(tgt_np, ypred_np, average='macro')
    # else:
    #     print("Only one class present in y_true. ROC AUC score is not defined.")
    #     auc = float('nan')  # Assign NaN if AUC calculation is not possible
    # auc = roc_auc_score(
    #     tgt_np,
    #     softmax_array,
    #     multi_class="ovr",
    #     average="macro",
    # )
        # Compute AUC
    try:
        auc = roc_auc_score(tgt_np, softmax_array[:,1]) #, multi_class="ovr", average="macro"
    except ValueError as e:
        print(f"Error computing AUC: {e}")
        auc = float('nan')  # Assign NaN if AUC calculation is not possible

    mastrix = {
        'accs': accs,
        # 'recalls': recalls,
        'g_mean': g_mean,
        'auc': auc,
        'f1_scores': f1_scores
    }
    return mastrix


def model_eval(i_epoch, data, model, args, ce_loss,device='cpu'):
    model.eval()
    with torch.no_grad():
        losses,losses1,losses2,losses3,losses4 = [],[],[],[],[]
        matrix_all={
        'accs':[],
        'g_mean':[],
        'auc':[],
        'f1_scores':[]
    }
        matrix1=deepcopy(matrix_all)
        matrix2=deepcopy(matrix_all)
        matrix3=deepcopy(matrix_all)
        matrix4=deepcopy(matrix_all)
        for intputx,label in data:
            intputx = intputx.to(device)
            label = label.to(device)
            y_,y_tsk1,y_tsk2,y_tsk3,y_tsk4 = model(intputx)
            loss_att = ce_loss(y_,label)
            loss_tsk1 = ce_loss(y_tsk1,label)
            loss_tsk2 = ce_loss(y_tsk2,label)
            loss_tsk3 = ce_loss(y_tsk3,label)
            loss_tsk4 = ce_loss(y_tsk4,label)
            loss = loss_att+0.01*(loss_tsk1+loss_tsk2+loss_tsk3+loss_tsk4)
            #loss = loss_att#+0.01*(loss_tsk1+loss_tsk2+loss_tsk3+loss_tsk4)
            losses.append(loss.item())
            losses1.append(loss_tsk1.item())
            losses2.append(loss_tsk2.item())
            losses3.append(loss_tsk3.item())
            losses4.append(loss_tsk4.item())
            y_att = torch.argmax(y_,dim=-1)
            y_1 = torch.argmax(y_tsk1,dim=-1)
            y_2 = torch.argmax(y_tsk2,dim=-1)
            y_3 = torch.argmax(y_tsk3,dim=-1)
            y_4 = torch.argmax(y_tsk4,dim=-1)
            m = compute_matrix(y_att,label,y_)
            m1 = compute_matrix(y_1, label,y_tsk1)
            m2 = compute_matrix(y_2,label,y_tsk2)
            m3 = compute_matrix(y_3,label,y_tsk3)
            m4 = compute_matrix(y_4,label,y_tsk4)
            for key in list(m1.keys()):
                matrix_all[key].append(m[key])
                matrix1[key].append(m1[key])
                matrix2[key].append(m2[key])
                matrix3[key].append(m3[key])
                matrix4[key].append(m4[key])
        loss_all = [losses,losses1,losses2,losses3,losses4]
    return loss_all,matrix_all,matrix1,matrix2,matrix3,matrix4

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    cls_num={
        # 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6 # for shuttle
        # 3:0, 4:1,5:2, 6:3, 7:4, 8:5  #for redwine
        # 'tested_negative':0, 'tested_positive':1  #for pima
        # 0:0, 1:1  #for creditcard
        # 3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6 # for whitewine
        # 0:0, 1:1, 2:2, 3:3, 4:4, 5:5  #for flare
        # 'g':0, 'h':1  #for magic
        #  -1:0, 1:1  #for titanic
        # ' van ':0, ' saab':1, ' bus ':2, ' opel':3 #for vehicle
        # 1:0, 2:1, 3:2, 4:3, 5:4 #for page-blocks
        # 'M':0, 'B':1  #for WDBC
        # 0:0, 1:1  #for muskv2
        #  -1:0, 1:1  #for gisette
        #  -1:0, 1:1  #for colon
        # 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9  #for fasionmnist
        # 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11,13:12,14:13,15:14,16:15,17:16,18:17,19:18,20:19,21:20,22:21,23:22,24:23,25:24,26:25  #for fasionmnist
        # 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11 #hapt
        # 'negative':0, 'positive':1  #for qsar
        # 'ad.':0, 'nonad.':1  #for ad
        -1:0,1:1 #acrene
        # 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11,13:12,14:13,15:14 #libras
        # -1:0, 1:1  #for madelon
    }
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    train_losses=[]
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf
    device = torch.device('cpu')
    ce_loss = torch.nn.CrossEntropyLoss()
    model = TSK_ATT(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    def lambda1(epoch_num): return 0.95 ** epoch_num
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    train_loader = DataLoader(
        AlignedConcDataset_1d(args, data_dir=args.data_train,cls_num=cls_num),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers)
    test_loader = DataLoader(
            AlignedConcDataset_1d(args, data_dir=args.data_val,cls_num=cls_num),
            batch_size=args.batch_sz, #args.batch_sz  shuttle 11600 / redwine 320/pima 154/credit 56962/whitewine 
            shuffle=False,
            num_workers=args.n_workers)
    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    # tlog = SummaryWriter(args.savedir)#tensorboard --logdir logs
    for i_epoch in range(start_epoch, args.max_epochs):
            train_losses = []
            model.train()
            optimizer.zero_grad()
            for inputx,label in tqdm(train_loader, total=len(train_loader)):
                inputx = inputx.to(device)
                label = label.to(device)
                y_,y_tsk1,y_tsk2,y_tsk3,y_tsk4 = model(inputx)
                loss_att = ce_loss(y_,label)
                loss_tsk1 = ce_loss(y_tsk1,label)
                loss_tsk2 = ce_loss(y_tsk2,label)
                loss_tsk3 = ce_loss(y_tsk3,label)
                loss_tsk4 = ce_loss(y_tsk4,label)
                loss = loss_att+0.01*(loss_tsk1+loss_tsk2+loss_tsk3+loss_tsk4)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                y_att = torch.argmax(y_,dim=-1)
                # print(y_)
                # print(y_att)
                # print(label)
                # m = compute_matrix(y_att,label,y_) #训练的时候不计算各个指标

            model.eval()
            loss_all_,matrix_all_,matrix1_,matrix2_,matrix3_,matrix4_ = model_eval(
                np.inf, test_loader, model, args, ce_loss
            )
            loss_all=[]
            matrix_all, matrix1,matrix2,matrix3,matrix4 ={},{},{},{},{}
            for num in range(len(loss_all_)):
                loss_all.append(np.mean(loss_all_[num]))
            for key in list(matrix1_.keys()):
                #print((matrix_all_[key]))
                matrix_all[key] = np.mean(list(matrix_all_[key]))
                matrix1[key]=np.mean(list(matrix1_[key]))
                matrix2[key]=np.mean(list(matrix2_[key]))
                matrix3[key]=np.mean(list(matrix3_[key]))
                matrix4[key]=np.mean(list(matrix4_[key]))
                
            logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
            #log_metrics("val", metrics, logger)
            logger.info(
                "{}:  all_loss: {:.5f}, tsk1_loss: {:.5f}, tsk2_loss: {:.5f}, tsk3_loss: {:.5f}, tsk4_loss: {:.5f}".format(
                    "val_loss", loss_all[0], loss_all[1],loss_all[2], loss_all[3],loss_all[4]
                )
            )
            logger.info(
                "{}: all_acc: {:.5f}, tsk1_acc: {:.5f}, tsk2_acc: {:.5f}, tsk3_acc: {:.5f}, tsk4_acc: {:.5f}".format(
                    "val_acc", matrix_all['accs'], matrix1['accs'],matrix2['accs'],matrix3['accs'],matrix4['accs'] 
                )
            )
            logger.info(
                "{}: all_g_mean: {:.5f}, tsk1_g_mean: {:.5f}, tsk2_g_mean: {:.5f}, tsk3_g_mean: {:.5f}, tsk4_g_mean: {:.5f}".format(
                    "val_g_mean",matrix_all['g_mean'], matrix1['g_mean'],matrix2['g_mean'],matrix3['g_mean'],matrix4['g_mean'] 
                )
            )
            logger.info(
                "{}:  all_auc: {:.5f}, tsk1_auc: {:.5f}, tsk2_auc: {:.5f}, tsk3_auc: {:.5f}, tsk4_auc: {:.5f}".format(
                    "val_auc", matrix_all['auc'], matrix1['auc'],matrix2['auc'],matrix3['auc'],matrix4['auc'] 
                )
            )
            logger.info(
                "{}:  all_f1: {:.5f}, tsk1_f1: {:.5f}, tsk2_f1: {:.5f}, tsk3_f1: {:.5f}, tsk4_f1: {:.5f}".format(
                    "val_f1", matrix_all['f1_scores'], matrix1['f1_scores'],matrix2['f1_scores'],matrix3['f1_scores'],matrix4 ['f1_scores']
                )
            )
            tuning_metric = matrix_all['f1_scores']
            # tlog.add_scalars('loss', {'train_loss':np.mean(train_losses),
            #                                 'val_loss':loss_all[0],
            #                             }, i_epoch)
            # tlog.add_scalars('loss_tsk', {'val_tsk1_loss': loss_all[1],
            #                                 'val_tsk2_loss': loss_all[2],
            #                                 'val_tsk3_loss':loss_all[3],
            #                                 'val_tsk4_loss':loss_all[4]
            #                             }, i_epoch)
            # tlog.add_scalars('val_acc', {'all_acc': matrix_all['accs'],
            #                             'tsk1_acc': matrix1['accs'],
            #                                 'tsk2_acc': matrix2['accs'],
            #                                 'tsk3_acc':matrix3['accs'],
            #                                 'tsk4_acc':matrix4['accs']
            #                             }, i_epoch)
            # tlog.add_scalars('val_g_mean', {'all_g_mean': matrix_all['g_mean'],
            #                                 'tsk1_g_mean': matrix1['g_mean'],
            #                                 'tsk2_g_mean':  matrix2['g_mean'],
            #                                 'tsk3_g_mean': matrix3['g_mean'],
            #                                 'tsk4_g_mean': matrix4['g_mean']
            #                             }, i_epoch)
            # tlog.add_scalars('val_auc', {'all_auc': matrix_all['auc'],
            #                             'tsk1_auc': matrix1['auc'],
            #                                 'tsk2_auc': matrix2['auc'],
            #                                 'tsk3_auc':matrix3['auc'],
            #                                 'tsk4_auc':matrix4['auc']
            #                             }, i_epoch)
            # tlog.add_scalars('val_f1', {'all_f1': matrix_all['f1_scores'],
            #                             'tsk1_f1': matrix1['f1_scores'],
            #                                 'tsk2_f1': matrix2['f1_scores'],
            #                                 'tsk3_f1':matrix3['f1_scores'],
            #                                 'tsk4_f1':matrix4['f1_scores']
            #                             }, i_epoch)
            scheduler.step(tuning_metric)
            is_improvement = tuning_metric > best_metric
            if is_improvement:
                best_metric = tuning_metric
                n_no_improve = 0
                save_checkpoint(
                {
                    "epoch": i_epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "n_no_improve": n_no_improve,
                    "best_metric": best_metric,
                },
                is_improvement,
                args.savedir,
                )
            else:
                n_no_improve += 1

            

            if n_no_improve >= args.patience:
                logger.info("No improvement. Breaking out of loop.")
                break
def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=64) #vehicle 32 / gisette 64/colon 4
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--data_train", type=str, default="data//now/acrene//train.csv")
    parser.add_argument("--data_val", type=str, default="data//now//acrene//val.csv")
    parser.add_argument("--max_seq_length", type=int, default=20)
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--input_dim", type=int, default=10000) #wdbc 30/shuttle 9 / redwine 11/pima 8/30 creditcard/ whitewine 11/flare 11/magic 10/titanic 3/vehicle 18/page-blocks 10/muskv2 166/gisette 5000/colon 2000/784 fashione/hapt 561/qsar 1024/ad 1558/acrene 10000/libras 90/madelon 500
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--img_hidden_sz", type=int, default=1000)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="acrene_yes_all_loss12")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default=".//savepath//tskatt//acrene/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2) #redwine 6 /shuttle7 /pima 2/whitewine 7/flare 6/magic 2/titanic 2/vehicle 4/page-blocks 5/muskv2 2/fashion 10/hapt 12/qsar 2/madelon 2
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--n_rule", type=int, default=5)
    parser.add_argument("--order", type=int, default=1)
def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()