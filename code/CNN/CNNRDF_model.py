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
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from util.comm import comm


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
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")

    device = torch.device(local_rank)

    writer = SummaryWriter('log')

    best_loss = 1000000
    seed_value = 35753

    torch.cuda.manual_seed_all(seed_value)

    dataset = RDFDataset('/data/jsliao/GNN/GTC_n18', device=device)

    data_size = len(dataset)
    print(data_size)
    batch_size = 4
    epochs = 1000

    trainsize = int(data_size * 0.8)
    validsize = int(data_size * 0.1)
    testsize = data_size - trainsize - validsize

    length = [trainsize, validsize, testsize]
    train_dataset, valid_dataset, test_dataset = random_split(dataset, length,
                                                              generator=torch.Generator().manual_seed(35753))

    sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    sampler_valid = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train,
                                               shuffle=False, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=sampler_valid,
                                               shuffle=False)

    CNN_model = CNN().to(device)
    model = torch.nn.parallel.DistributedDataParallel(CNN_model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0., nesterov=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-7, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.999,last_epoch=-1)

    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        loss_sum_train = 0
        list_of_trloss = [torch.zeros(1).to(device) for _ in range(4)]
        train_loader.sampler.set_epoch(epoch)
        for rdf, y_tc in train_loader:
            optimizer.zero_grad()
            # torch.cuda.empty_cache()
            pred_tc = model(rdf)
            # torch.cuda.empty_cache()
            loss = criterion(pred_tc, y_tc)
            loss_sum_train = loss_sum_train + loss
            # 添加L1正则化
            l1_lambda = 1e-6
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

            loss.backward()
            # torch.cuda.empty_cache()
            optimizer.step()
            # torch.cuda.empty_cache()
            # print(f'Epoch: {epoch:03d}, Train loss: {loss:.4f}')
        avg_train_loss = loss_sum_train / len(train_loader)
        # torch.cuda.empty_cache()
        writer.add_scalar('train_loss', avg_train_loss, epoch)
        # torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.all_gather(list_of_trloss, avg_train_loss)  # 收集所有进程的输出到list_of_outputs中
        average_trloss = torch.mean(torch.stack(list_of_trloss), dim=0)
        writer.add_scalar('train_loss_average', average_trloss, epoch)

        loss_sum_test = 0
        # torch.cuda.empty_cache()
        model.eval()
        list_of_loss = [torch.zeros(1).to(device) for _ in range(4)]
        for rdf, y_tc in valid_loader:
            test_tc = model(rdf)
            # torch.cuda.empty_cache()
            loss = criterion(test_tc, y_tc)
            # torch.cuda.empty_cache()
            loss_sum_test = loss_sum_test + loss
            # torch.cuda.empty_cache()
            # print(f'Epoch: {epoch:03d}, Test loss: {loss:.4f}')

        avg_test_loss = torch.tensor(loss_sum_test / len(valid_loader)).to(device)
        # torch.cuda.empty_cache()
        writer.add_scalar('test_loss', avg_test_loss, epoch)
        comm.synchronize()

        scheduler.step()
        # torch.cuda.empty_cache()
        dist.all_gather(list_of_loss, avg_test_loss)  # 收集所有进程的输出到list_of_outputs中
        # print(list_of_loss)
        average_loss = torch.mean(torch.stack(list_of_loss), dim=0)
        # print(average_loss)
        writer.add_scalar('test_loss_average', average_loss, epoch)

        # final_loss = (average_loss.item() + average_trloss.item()) / 2

        if average_loss.item() <= best_loss and comm.is_main_process():
            torch.save(model.module.state_dict(), 'best_gtc_gat.pth')
            best_loss = average_loss.item()

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('lr', lr, epoch)

        print(
            f'RANK: {local_rank}, Epoch: {epoch:03d}, Train loss: {avg_train_loss:.4f}({average_trloss.item():.4f}), Test loss: {avg_test_loss:.4f}({average_loss.item():.4f}), lr: {lr}')
        if comm.is_main_process():
            torch.save(model.module.state_dict(), 'last_gtc_gat.pth')
        # torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    main()

