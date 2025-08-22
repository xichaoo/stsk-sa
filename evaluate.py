import torch
from torch.utils.data import DataLoader
import numpy as np
from model import TSK_ATT
from main_layersEX import AlignedConcDataset_1d, compute_matrix, compute_g_mean  # 确保这些函数被导入
# from main import AlignedConcDataset_1d, compute_matrix, compute_g_mean  # 确保这些函数被导入
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="savepath/tskatt/hapt/hapt_yes_all_loss1_8layers/checkpoint.pt")
    parser.add_argument("--test_csv", type=str, default="data/now/hapt/val.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=560) # qsar 1024 acrene 10000 hapt 560 fashionemnist 783
    parser.add_argument("--max_seq_length", type=int, default=20)  # 加这一行！
    parser.add_argument("--n_classes", type=int, default=12)
    parser.add_argument("--n_rule", type=int, default=5)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--mode", type=int, default=2)
    # parser.add_argument("--input_dim", type=int, default=1024) #wdbc 30/shuttle 9 / redwine 11/pima 8/30 creditcard/ whitewine 11/flare 11/magic 10/titanic 3/vehicle 18/page-blocks 10/muskv2 166/gisette 5000/colon 2000/783 fashione/hapt 561/qsar 1024/ad 1558/acrene 10000/libras 90/madelon 500
    # parser.add_argument("--dropout", type=float, default=0.1)
    # parser.add_argument("--img_hidden_sz", type=int, default=1000)
    # parser.add_argument("--include_bn", type=int, default=True)
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--lr_factor", type=float, default=0.3)
    # parser.add_argument("--lr_patience", type=int, default=10)
    # parser.add_argument("--max_epochs", type=int, default=300)
    # parser.add_argument("--n_workers", type=int, default=4)
    # parser.add_argument("--name", type=str, default="qsar_yes_all_loss1_5layers")
    # parser.add_argument("--patience", type=int, default=20)
    # parser.add_argument("--savedir", type=str, default=".//savepath//tskatt//qsar/")
    # parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--n_classes", type=int, default=2) #redwine 6 /shuttle7 /pima 2/whitewine 7/flare 6/magic 2/titanic 2/vehicle 4/page-blocks 5/muskv2 2/fashion 10/hapt 12/qsar 2/madelon 2
    # parser.add_argument("--annealing_epoch", type=int, default=10)
    # parser.add_argument("--n_rule", type=int, default=5)
    # parser.add_argument("--order", type=int, default=1)
    return parser.parse_args(args=[])

def main():
    args = get_args()

    # 加载模型和参数
    model = TSK_ATT(args)
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    model.eval()

    # 标签映射 (根据训练时保持一致)
    # cls_num = {'negative': 0, 'positive': 1}
    # cls_num = {-1:0, 1:1} #acrene
    cls_num = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11} #hapt
    # cls_num = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}  #for fasionmnist

    # 加载测试集
    test_loader = DataLoader(
        AlignedConcDataset_1d(args, data_dir=args.test_csv, cls_num=cls_num),
        batch_size=args.batch_size,
        shuffle=False
    )

    # 损失函数
    ce_loss = torch.nn.CrossEntropyLoss()

    # 模型评估
    from main_layersEX import model_eval  # 确保 model_eval 可导入
    # from main import model_eval  # 确保 model_eval 可导入

    _, matrix_all, *_ = model_eval(0, test_loader, model, args, ce_loss, device=args.device)

    # 输出平均指标
    results = {k: np.mean(matrix_all[k]) for k in matrix_all}
    print("=== Final Test Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
