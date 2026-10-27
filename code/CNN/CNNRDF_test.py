import sys

sys.path.append('/data/jsliao/GNN')

import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def UnMaxMinNormalization(data,min,max):
    return data*(max-min) + min

def MaxMinNormalization(data,min,max):
    data = (data - min) / (max - min)
    return data

class RDFDataset(Dataset):
    def __init__(self, file_path, device):
        # 读取CSV文件
        self.data_rdf = pd.read_csv(os.path.join(file_path, 'GTC/raw/rdf_y.txt'), sep=',', header=None)
        self.data_tc = pd.read_csv(os.path.join(file_path, 'GTC/raw/GTC_graph_attributes.txt'), sep=',', header=None)

        self.df11 = self.data_rdf.iloc[:, :100]

        self.df12 = self.data_rdf.iloc[:, 100:200]

        self.df13 = self.data_rdf.iloc[:, 200:300]

        self.df22 = self.data_rdf.iloc[:, 400:500]

        self.df23 = self.data_rdf.iloc[:, 500:600]

        self.df33 = self.data_rdf.iloc[:, 800:900]

        self.xdata = pd.concat((self.df11, self.df12, self.df13, self.df22, self.df23, self.df33), axis=1)
        self.xdata = self.xdata / 14

        self.data = pd.concat((self.xdata, self.data_tc), axis=1)

        self.device = device

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 使用.iloc明确基于位置索引
        row = self.data.iloc[idx]
        # 将特征和标签分开
        features = torch.tensor(row.iloc[:-1].to_numpy(), dtype=torch.float32)  # 特征
        label = torch.tensor(row.iloc[-1], dtype=torch.float32)  # 标签
        return features.to(self.device), label.to(self.device)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一层卷积模块
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 第二层卷积模块
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 第三层卷积模块
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(16)  # 自适应池化到固定长度
        )

        # MLP部分
        self.mlp = nn.Sequential(
            nn.Linear(32 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 输入形状: (batch_size, 600)
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch_size, 1, 600)

        # 卷积部分
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 展平
        x = x.view(x.size(0), -1)

        # MLP部分
        x = self.mlp(x)

        return x.squeeze(-1)


def main():
    seed_value = 35753
    device = torch.device('cuda')
    best_file = 'best_gtc_gat.pth'
    last_file = 'last_gtc_gat.pth'

    torch.cuda.manual_seed_all(seed_value)

    dataset = RDFDataset('/data/jsliao/GNN/GTC_n18', device=device)

    data_size = len(dataset)
    print(data_size)
    batch_size = 16

    trainsize = int(data_size * 0.8)
    validsize = int(data_size * 0.1)
    testsize = data_size - trainsize - validsize

    length = [trainsize, validsize, testsize]
    train_dataset, valid_dataset, test_dataset = random_split(dataset, length,
                                                              generator=torch.Generator().manual_seed(35753))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN().to(device)

    state_dict = torch.load(best_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    criterion = torch.nn.MSELoss()

    model.eval()
    trpredTc = []
    trtargetTc = []
    predTc = []
    targetTc = []
    tepredTc = []
    tetargetTc = []
    for rdf, y_tc in train_loader:
        test_tc = model(rdf)
        loss = criterion(test_tc, y_tc)
        trtargetTc = np.append(trtargetTc, y_tc.cpu().detach().numpy())
        trpredTc = np.append(trpredTc, test_tc.cpu().detach().numpy())
        print('trvalid :', loss)
        # comm.synchronize()
    for rdf, y_tc in valid_loader:
        test_tc = model(rdf)
        loss = criterion(test_tc, y_tc)
        targetTc = np.append(targetTc, y_tc.cpu().detach().numpy())
        predTc = np.append(predTc, test_tc.cpu().detach().numpy())
        print('valid :', loss)
    for rdf, y_tc in test_loader:
        test_tc = model(rdf)
        loss = criterion(test_tc, y_tc)
        tetargetTc = np.append(tetargetTc, y_tc.cpu().detach().numpy())
        tepredTc = np.append(tepredTc, test_tc.cpu().detach().numpy())
        print('tevalid :', loss)
        # comm.synchronize()

    trtargetTc = trtargetTc * 1800
    trpredTc = trpredTc * 1800

    targetTc = targetTc * 1800
    predTc = predTc * 1800

    tetargetTc = tetargetTc * 1800
    tepredTc = tepredTc * 1800

    x45 = np.linspace(0, 1800.0, 10)
    y45 = x45

    x45n = np.linspace(0, 1, 100)
    y45n = x45n

    mae = mean_absolute_error(tetargetTc, tepredTc)
    rmse = np.sqrt(mean_squared_error(tetargetTc, tepredTc))
    r_2 = r2_score(tetargetTc, tepredTc)
    print(mae, rmse, r_2)

    plt.figure(figsize=(5, 5))
    plt.plot(x45, y45, c='r', linewidth=0.5)
    plt.scatter(trtargetTc, trpredTc, c='dodgerblue', alpha=0.7, s=8, edgecolors='blue', linewidths=0.3, label='Train')
    plt.scatter(targetTc, predTc, c='limegreen', s=8, edgecolors='green', linewidths=0.3, label='Valid')
    plt.scatter(tetargetTc, tepredTc, c='red', s=8, edgecolors='firebrick', linewidths=0.3, label='Test')
    plt.text(-1, 1795, s=f"$R^2$={round(r_2, 2)}", fontsize=13)
    plt.text(-1, 1695, 'RMSE={:.2f}'.format(rmse), fontsize=13)
    plt.text(-1, 1595, 'MAE={:.2f}'.format(mae), fontsize=13)
    plt.xlabel('MD thermal conductivity (W/mK)')
    plt.ylabel('Predicted thermal conductivity (W/mK)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tc_test_GAT.png', dpi=600)
    # print(seed_value)

    plt.figure(figsize=(5, 5))
    plt.plot(x45n, y45n, c='r', linewidth=0.5)
    plt.scatter(MaxMinNormalization(trtargetTc, 0, 1700), MaxMinNormalization(trpredTc, 0, 1700), c='dodgerblue',
                alpha=0.7, s=8, edgecolors='blue', linewidths=0.3, label='Train')
    plt.scatter(MaxMinNormalization(targetTc, 0, 1700), MaxMinNormalization(predTc, 0, 1700), c='limegreen', s=8,
                edgecolors='green', linewidths=0.3, label='Valid')
    plt.scatter(MaxMinNormalization(tetargetTc, 0, 1700), MaxMinNormalization(tepredTc, 0, 1700), c='red', s=8,
                edgecolors='firebrick', linewidths=0.3, label='Test')
    plt.text(0.1, 0.9, s=f"$R^2$={round(r_2, 2)}", fontsize=13)
    plt.text(0.1, 0.8, 'RMSE={:.2f}'.format(rmse), fontsize=13)
    plt.text(0.1, 0.7, 'MAE={:.2f}'.format(mae), fontsize=13)
    plt.xlabel('MD thermal conductivity (W/mK)')
    plt.ylabel('Predicted thermal conductivity (W/mK)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tc_test_GAT_N.png', dpi=600)

if __name__ == "__main__":
    main()

