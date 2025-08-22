from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.manifold import TSNE
import matplotlib as mpl

### ---------- for hapt dataset-------------------
# test = pd.read_csv('data/now/hapt/val.csv')
# X_test = test.iloc[:, 1:-1]   # 前 784 列：像素值
# y      = test.iloc[:,  -1]   # 第 785 列：标签
# print(X_test.shape)  # (10000, 784)
# print(y.shape)       # (10000,)

# tsne = TSNE(n_components=2)
# train_test_2D = tsne.fit_transform(X_test)
# plt.figure(figsize=(6, 6))  # 控制物理尺寸
# plt.scatter(train_test_2D[:,0], train_test_2D[:,1], c=y, cmap='jet', s=0.5)
# plt.title("Hapt-Original", fontsize=14)
# # plt.xlabel("Component 1", fontsize=12)
# # plt.ylabel("Component 2", fontsize=12)
# plt.tick_params(labelsize=10)
# # plt.axis('off')
# # 设置坐标轴线宽更细
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_linewidth(0.3)
# plt.tight_layout()
# # 保存为高清图像
# plt.savefig("hapt_ori2.pdf", bbox_inches='tight')

test1 = pd.read_csv('test_Z_outputs_hapt.csv')
X_test1 = test1.iloc[:, :-1]   # 前 784 列：像素值
y1      = test1.iloc[:,  -1]   # 第 785 列：标签
print(X_test1.shape)  # (10000, 784)
print(y1.shape)       # (10000,)
tsne1 = TSNE(n_components=2)
train_test_2D1 = tsne1.fit_transform(X_test1)
plt.figure(figsize=(6, 6))  # 控制物理尺寸
# 自定义蓝红色调色盘
# cmap = ListedColormap(['blue', 'red'])
plt.scatter(train_test_2D1[:,0], train_test_2D1[:,1], c=y1, cmap='jet', s=0.5)
plt.title("Hapt - Z", fontsize=14)
# plt.xlabel("Component 1", fontsize=12)
# plt.ylabel("Component 2", fontsize=12)
plt.tick_params(labelsize=10)
# plt.axis('off')
# 设置坐标轴线宽更细
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(0.3)
plt.tight_layout()
# 保存为高清图像
plt.savefig("hapt_z3.pdf", bbox_inches='tight')

# ### ---------- for QSAR dataset-------------------
# test = pd.read_csv('data/now/qsar/val.csv')
# X_test = test.iloc[:, 1:-1]   # 前 784 列：像素值
# y      = test.iloc[:,  -1]   # 第 785 列：标签
# print(X_test.shape)  # (10000, 784)
# print(y.shape)       # (10000,)
# label_map = {'negative': 0, 'positive': 1}
# y_numeric = y.map(label_map)
# tsne = TSNE(n_components=2)
# train_test_2D = tsne.fit_transform(X_test)
# plt.figure(figsize=(6, 6))  # 控制物理尺寸
# cmap = ListedColormap(['blue', 'red'])
# plt.scatter(train_test_2D[:,0], train_test_2D[:,1], c=y_numeric, cmap=cmap, s=0.5)
# plt.title("Qsar-Original", fontsize=14)
# # plt.xlabel("Component 1", fontsize=12)
# # plt.ylabel("Component 2", fontsize=12)
# plt.tick_params(labelsize=10)
# # plt.axis('off')
# # 设置坐标轴线宽更细
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_linewidth(0.3)
# plt.tight_layout()
# # 保存为高清图像
# plt.savefig("qsar_ori3.pdf", bbox_inches='tight')

# # test1 = pd.read_csv('test_Z_outputs_qsar.csv')
# # X_test1 = test1.iloc[:, :-1]   # 前 784 列：像素值
# # y1      = test1.iloc[:,  -1]   # 第 785 列：标签
# # print(X_test1.shape)  # (10000, 784)
# # print(y1.shape)       # (10000,)
# # # label_map = {'negative': 0, 'positive': 1}
# # # y_numeric = y.map(label_map)
# # tsne1 = TSNE(n_components=2)
# # train_test_2D1 = tsne1.fit_transform(X_test1)
# # plt.figure(figsize=(6, 6))  # 控制物理尺寸
# # # 自定义蓝红色调色盘
# # cmap = ListedColormap(['blue', 'red'])
# # plt.scatter(train_test_2D1[:,0], train_test_2D1[:,1], c=y1, cmap=cmap, s=0.5)
# # plt.title("Qsar - Z", fontsize=14)
# # # plt.xlabel("Component 1", fontsize=12)
# # # plt.ylabel("Component 2", fontsize=12)
# # plt.tick_params(labelsize=10)
# # # plt.axis('off')
# # # 设置坐标轴线宽更细
# # ax = plt.gca()
# # for spine in ax.spines.values():
# #     spine.set_linewidth(0.3)
# # plt.tight_layout()
# # # 保存为高清图像
# # plt.savefig("qsar_z3.pdf", bbox_inches='tight')

# # ### ---------- for arcene dataset-------------------

# test = pd.read_csv('data/now/acrene/val.csv')
# X_test = test.iloc[:, 1:-1]   # 前 784 列：像素值
# y      = test.iloc[:,  -1]   # 第 785 列：标签
# print(X_test.shape)  # (10000, 784)
# print(y.shape)       # (10000,)
# # label_map = {'negative': 0, 'positive': 1}
# # y_numeric = y.map(label_map)
# tsne = TSNE(n_components=2)
# train_test_2D = tsne.fit_transform(X_test)
# plt.figure(figsize=(6, 6))  # 控制物理尺寸
# cmap = ListedColormap(['blue', 'red'])
# plt.scatter(train_test_2D[:,0], train_test_2D[:,1], c=y, cmap=cmap, s=3)
# plt.title("Arcene-Original", fontsize=14)
# # plt.xlabel("Component 1", fontsize=12)
# # plt.ylabel("Component 2", fontsize=12)
# plt.tick_params(labelsize=10)
# # plt.axis('off')
# # 设置坐标轴线宽更细
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_linewidth(0.3)
# plt.tight_layout()
# # 保存为高清图像
# plt.savefig("arcene_ori3.pdf", bbox_inches='tight')

# # test1 = pd.read_csv('test_Z_outputs_arcene.csv')
# # X_test1 = test1.iloc[:, :-1]   # 前 784 列：像素值
# # y1      = test1.iloc[:,  -1]   # 第 785 列：标签
# # print(X_test1.shape)  # (10000, 784)
# # print(y1.shape)       # (10000,)
# # # label_map = {'negative': 0, 'positive': 1}
# # # y_numeric = y.map(label_map)
# # tsne1 = TSNE(n_components=2)
# # train_test_2D1 = tsne1.fit_transform(X_test1)
# # plt.figure(figsize=(6, 6))  # 控制物理尺寸
# # # 自定义蓝红色调色盘
# # cmap = ListedColormap(['blue', 'red'])
# # plt.scatter(train_test_2D1[:,0], train_test_2D1[:,1], c=y1, cmap=cmap, s=3)
# # plt.title("Arcene - Z", fontsize=14)
# # # plt.xlabel("Component 1", fontsize=12)
# # # plt.ylabel("Component 2", fontsize=12)
# # plt.tick_params(labelsize=10)
# # # plt.axis('off')
# # # 设置坐标轴线宽更细
# # ax = plt.gca()
# # for spine in ax.spines.values():
# #     spine.set_linewidth(0.3)
# # plt.tight_layout()
# # # 保存为高清图像
# # plt.savefig("arcene_z3.pdf", bbox_inches='tight')


# ## ---------- for fashinmnist dataset-------------------

# test = pd.read_csv('data/now/fasionmnist/val.csv')
# X_test = test.iloc[:, :-1]   # 前 784 列：像素值
# y      = test.iloc[:,  -1]   # 第 785 列：标签
# print(X_test.shape)  # (10000, 784)
# print(y.shape)       # (10000,)
# # label_map = {'negative': 0, 'positive': 1}
# # y_numeric = y.map(label_map)
# tsne = TSNE(n_components=2)
# train_test_2D = tsne.fit_transform(X_test)
# plt.figure(figsize=(6, 6))  # 控制物理尺寸
# plt.scatter(train_test_2D[:,0], train_test_2D[:,1], c=y, cmap='jet', s=0.5)
# plt.title("FashionMNIST-Original", fontsize=14)
# # plt.xlabel("Component 1", fontsize=12)
# # plt.ylabel("Component 2", fontsize=12)
# plt.tick_params(labelsize=10)
# # plt.axis('off')
# # 设置坐标轴线宽更细
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_linewidth(0.3)
# plt.tight_layout()
# # 保存为高清图像
# plt.savefig("fashionmnist_ori3.pdf", bbox_inches='tight')

# # # 设置全局坐标轴线条宽度
# # mpl.rcParams['axes.linewidth'] = 0.3

# # test1 = pd.read_csv('test_Z_outputs_fashionmnist.csv')
# # X_test1 = test1.iloc[:, :-1]   # 前 784 列：像素值
# # y1      = test1.iloc[:,  -1]   # 第 785 列：标签
# # print(X_test1.shape)  # (10000, 784)
# # print(y1.shape)       # (10000,)
# # # label_map = {'negative': 0, 'positive': 1}
# # # y_numeric = y.map(label_map)
# # tsne1 = TSNE(n_components=2)
# # train_test_2D1 = tsne1.fit_transform(X_test1)
# # plt.figure(figsize=(6, 6))  # 控制物理尺寸
# # plt.scatter(train_test_2D1[:,0], train_test_2D1[:,1], c=y1, cmap='jet', s=0.5)
# # plt.title("FashionMNIST - Z", fontsize=14)
# # # plt.xlabel("Component 1", fontsize=12)
# # # plt.ylabel("Component 2", fontsize=12)
# # plt.tick_params(labelsize=10)
# # # 设置坐标轴线宽更细
# # ax = plt.gca()
# # for spine in ax.spines.values():
# #     spine.set_linewidth(0.3)

# # plt.tight_layout()
# # # 保存为高清图像
# # plt.savefig("fashionmnist_z4.pdf", bbox_inches='tight')