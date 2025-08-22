import torch
from torch.utils.data import DataLoader
import pandas as pd
from model import TSK_ATT  # 假设模型定义在 model.py 中
from utils import set_seed  # 你已有的设置函数
# from your_data_module import AlignedConcDataset_1d  # 替换为你 Dataset 的实际位置
from torch.utils.data import Dataset
import argparse
import os
import numpy as np

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

def extract_Z_and_save(args):
    # 类别映射（与你训练时保持一致）
    # cls_num = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}  # 例如fashionmnist
    cls_num = {-1:0,1:1}  #acrene
    # cls_num = {-1:0, 1:1}  #for colon
    # cls_num = {'negative':0, 'positive':1} #for qsar
    # cls_num = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11} #hapt

    device = torch.device("cpu")
    set_seed(args.seed)

    # 加载验证集（或测试集）
    test_dataset = AlignedConcDataset_1d(args, data_dir=args.data_val, cls_num=cls_num)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = TSK_ATT(args)
    model.load_state_dict(torch.load(os.path.join(args.savedir, "checkpoint.pt"))["state_dict"])
    model.to(device)
    model.eval()

    # 存储所有样本的 Z 表征
    Z_list = []

    with torch.no_grad():
        for inputx, label in test_loader:
            inputx = inputx.to(device)
            _ = model(inputx)
            z = model.output_z.squeeze(0).cpu().numpy()
            label_val = label.item()  # 转成int
            z_with_label = np.append(z, label_val)  # 把标签拼接在末尾
            Z_list.append(z_with_label)

    Z_matrix = np.stack(Z_list)
    z_dim = Z_matrix.shape[1] - 1
    columns = [f"Z_{i}" for i in range(z_dim)] + ["label"]
    df = pd.DataFrame(Z_matrix, columns=columns)
    df.to_csv("test_Z_outputs.csv", index=False)

    print("Saved Z representations to test_Z_outputs.csv")

# 解析参数
def get_args(parser):
    parser.add_argument("--savedir", type=str, default=".//savepath//tskatt//acrene/acrene_yes_all_loss1/")
    parser.add_argument("--data_val", type=str, default="data//now//acrene//val.csv")
    parser.add_argument("--input_dim", type=int, default=10000) #784 fashionmnist/ 10000 arcene/ 2000 colon/hapt 561
    parser.add_argument("--batch_sz", type=int, default=64)
    parser.add_argument("--n_classes", type=int, default=2) # 10 fashionmnist /2 arcene/2colon/2 /hapt 12
    parser.add_argument("--n_rule", type=int, default=5)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--img_hidden_sz", type=int, default=1000)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--max_seq_length", type=int, default=20)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)

if __name__ == "__main__":
    args = torch.load(os.path.join(".//savepath//tskatt//acrene/acrene_yes_all_loss1/", "args.pt"))
    extract_Z_and_save(args)

