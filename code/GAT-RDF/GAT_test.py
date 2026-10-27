import sys
sys.path.append('/data/jsliao/GNN')

from GTC import GTCDATASET
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.datasets import planetoid
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
# from util.comm import comm

import torch.distributed as dist

def UnMaxMinNormalization(data,min,max):
    return data*(max-min) + min

def MaxMinNormalization(data,min,max):
    data = (data - min) / (max - min)
    return data

class GAT(nn.Module):
    def __init__(self, num_node_features):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=8, edge_dim=7, heads=4)
        self.convf = GATv2Conv(in_channels=4 * 8, out_channels=16, edge_dim=7, heads=1)
        self.layer_res0 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(True),
                                        nn.Linear(32, 32), nn.ReLU(True),
                                        nn.Linear(32, 64), nn.ReLU(True),
                                        nn.Linear(64, 64), nn.ReLU(True))
        self.out = nn.Sequential(nn.Linear(64, 1))
        self.layer_res0.apply(self.weight_init)
        self.out.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            # nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            print('useing weigh_init linear')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            print('useing weigh_init bn1d')

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.convf(x, edge_index, edge_attr)
        graph_embedding = global_max_pool(x, batch).squeeze(-1)
        x = self.layer_res0(graph_embedding)
        x = self.out(x).squeeze(-1)
        return x

def main():
    # local_rank = int(os.environ["LOCAL_RANK"])
    # dist.init_process_group(backend="nccl")

    # device = torch.device(local_rank)
    # GTC_directory = '/data/jsliao/GNN/GTC_n18/GTC/raw'  # F:\\GTC_n8\\GTC\\raw
    # seed_value = random.randint(1, 10000)
    seed_value = np.random.randint(1, 10000)
    device = torch.device('cuda')
    best_file = 'best_gtc_gat.pth'
    last_file = 'last_gtc_gat.pth'

    torch.cuda.manual_seed_all(seed_value)
    # max = pd.read_csv(os.path.join(GTC_directory, 'MAX.txt'), sep=',', header=None)
    # max = max.values
    # max = torch.tensor(max, dtype=torch.float).to(device)
    # min = pd.read_csv(os.path.join(GTC_directory, 'MIN.txt'), sep=',', header=None)
    # min = min.values
    # min = torch.tensor(min, dtype=torch.float).to(device)
    # dataset = GTCDATASET(root='C:\\Users\\17403\\PycharmProjects\\TRANSCRY\\GNN\\GTC', name='GTC', use_node_attr=True, use_edge_attr=True)
    dataset = GTCDATASET(root='/data/jsliao/GNN/GTC_n18_rdf', name='GTC', use_node_attr=True, use_edge_attr=True)
    print(dataset)
    dataset = dataset.to(device)

    data_size = len(dataset)
    print(data_size)
    batch_size = 32

    trainsize = int(data_size*0.8)
    validsize = int(data_size*0.1)
    testsize = data_size - trainsize - validsize

    length = [trainsize, validsize, testsize]
    train_dataset, valid_dataset, test_dataset= random_split(dataset, length, generator=torch.Generator().manual_seed(35753))

    # sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    # sampler_test = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=True)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train, shuffle=False)
    # valid_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler_test, shuffle=False)
    #
    # GAT_model = GAT(num_node_features=dataset.data.x.shape[1]).to(device)
    # model = torch.nn.parallel.DistributedDataParallel(GAT_model,
    #                                                   device_ids=[local_rank],
    #                                                   output_device=local_rank,
    #                                                   find_unused_parameters=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GAT(num_node_features=dataset.data.x.shape[1])

    state_dict = torch.load(best_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.SmoothL1Loss()

    model.eval()
    trpredTc = []
    trtargetTc = []
    predTc = []
    targetTc = []
    tepredTc = []
    tetargetTc = []
    for data in train_loader:
        test_tc = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(test_tc, data.y)
        trtargetTc = np.append(trtargetTc, data.y.cpu().detach().numpy())
        trpredTc = np.append(trpredTc, test_tc.cpu().detach().numpy())
        print('trvalid :', loss)
        # comm.synchronize()
    for data in valid_loader:
        test_tc = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(test_tc, data.y)
        targetTc = np.append(targetTc, data.y.cpu().detach().numpy())
        predTc = np.append(predTc, test_tc.cpu().detach().numpy())
        print('valid :', loss)
    for data in test_loader:
        test_tc = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(test_tc, data.y)
        tetargetTc = np.append(tetargetTc, data.y.cpu().detach().numpy())
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
    plt.scatter(MaxMinNormalization(trtargetTc, 0, 1700), MaxMinNormalization(trpredTc, 0, 1700), c='dodgerblue', alpha=0.7, s=8, edgecolors='blue', linewidths=0.3, label='Train')
    plt.scatter(MaxMinNormalization(targetTc, 0, 1700), MaxMinNormalization(predTc, 0, 1700), c='limegreen', s=8, edgecolors='green', linewidths=0.3, label='Valid')
    plt.scatter(MaxMinNormalization(tetargetTc, 0, 1700), MaxMinNormalization(tepredTc, 0, 1700), c='red', s=8, edgecolors='firebrick', linewidths=0.3, label='Test')
    plt.text(0.1, 0.9, s=f"$R^2$={round(r_2, 2)}", fontsize=13)
    plt.text(0.1, 0.8, 'RMSE={:.2f}'.format(rmse), fontsize=13)
    plt.text(0.1, 0.7, 'MAE={:.2f}'.format(mae), fontsize=13)
    plt.xlabel('MD thermal conductivity (W/mK)')
    plt.ylabel('Predicted thermal conductivity (W/mK)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tc_test_GAT_N.png', dpi=600)
    # state_dict = torch.load(last_file, map_location="cpu")
    # model.load_state_dict(state_dict)
    # model.to(device)
    #
    # model.eval()
    # trpredTc = []
    # trtargetTc = []
    # predTc = []
    # targetTc = []
    # for data in train_loader:
    #     test_tc = model(data.x, data.edge_index, data.edge_attr, data.batch)
    #     loss = criterion(test_tc, data.y)
    #     trtargetTc = np.append(trtargetTc, data.y.cpu().detach().numpy())
    #     trpredTc = np.append(trpredTc, test_tc.cpu().detach().numpy())
    #     print('trvalid :', loss)
    #     # comm.synchronize()
    # for data in valid_loader:
    #     test_tc = model(data.x, data.edge_index, data.edge_attr, data.batch)
    #     loss = criterion(test_tc, data.y)
    #     targetTc = np.append(targetTc, data.y.cpu().detach().numpy())
    #     predTc = np.append(predTc, test_tc.cpu().detach().numpy())
    #     print('valid :', loss)
    #     # comm.synchronize()
    #
    # trtargetTc = UnMaxMinNormalization(trtargetTc, min.cpu().detach().numpy().reshape(1),
    #                                    max.cpu().detach().numpy().reshape(1))
    # trpredTc = UnMaxMinNormalization(trpredTc, min.cpu().detach().numpy().reshape(1), max.cpu().detach().numpy().reshape(1))
    #
    # targetTc = UnMaxMinNormalization(targetTc, min.cpu().detach().numpy().reshape(1), max.cpu().detach().numpy().reshape(1))
    # predTc = UnMaxMinNormalization(predTc, min.cpu().detach().numpy().reshape(1), max.cpu().detach().numpy().reshape(1))
    #
    # x45 = np.linspace(0, 2500.0, 10)
    # y45 = x45
    #
    # mae = mean_absolute_error(targetTc, predTc)
    # rmse = np.sqrt(mean_squared_error(targetTc, predTc))
    # r_2 = r2_score(targetTc, predTc)
    # print(mae, rmse, r_2)
    #
    # plt.figure(figsize=(5, 5))
    # plt.plot(x45, y45, c='r', linewidth=0.5)
    # plt.scatter(trtargetTc, trpredTc, c='b', alpha=0.7, s=5)
    # plt.scatter(targetTc, predTc, c='gold', alpha=0.7, s=5)
    # plt.text(-1, 2495, s=f"$R^2$={round(r_2, 2)}", fontsize=13)
    # plt.text(-1, 2395, 'RMSE={:.2f}'.format(rmse), fontsize=13)
    # plt.text(-1, 2295, 'MAE={:.2f}'.format(mae), fontsize=13)
    # plt.tight_layout()
    # plt.savefig('tc_test_GAT_la.png', dpi=300)
    # print(seed_value)

if __name__ == "__main__":
    main()