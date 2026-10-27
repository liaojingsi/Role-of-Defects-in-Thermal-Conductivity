import sys
sys.path.append('/data/jsliao/GNN')

from GTC import GTCDATASET
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.datasets import planetoid
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from util.comm import comm

import torch.distributed as dist

def graph_data_augmentation(batch, augmentation_prob=0.3):
    """图数据增强"""
    if torch.rand(1) > augmentation_prob:
        return batch

    augmented_batch = batch.clone()

    # 1. 节点特征噪声
    noise_std = 0.02 * augmented_batch.x.std()
    augmented_batch.x = augmented_batch.x + torch.randn_like(augmented_batch.x) * noise_std

    # 2. 边特征噪声
    if augmented_batch.edge_attr is not None:
        edge_noise_std = 0.02 * augmented_batch.edge_attr.std()
        augmented_batch.edge_attr = augmented_batch.edge_attr + \
                                    torch.randn_like(augmented_batch.edge_attr) * edge_noise_std

    return augmented_batch

class LabelSmoothMSELoss(nn.Module):
    """回归任务的标签平滑"""
    def __init__(self, smoothing=0.05):
        super(LabelSmoothMSELoss, self).__init__()
        self.smoothing = smoothing
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        if self.training and self.smoothing > 0:
            # 在训练时对目标添加少量噪声
            noise = torch.randn_like(target) * self.smoothing * target.std()
            smooth_target = target + noise
            return self.mse(pred, smooth_target)
        else:
            return self.mse(pred, target)

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
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")

    device = torch.device(local_rank)

    writer = SummaryWriter('log')

    best_loss = 1000000
    seed_value = np.random.randint(1, 10000)

    torch.cuda.manual_seed_all(seed_value)
    dataset = GTCDATASET(root='/data/jsliao/GNN/GTC_n18_rdf', name='GTC', use_node_attr=True, use_edge_attr=True)
    print(dataset)
    dataset = dataset.to(device)

    data_size = len(dataset)
    print(data_size)
    batch_size = 16
    epochs = 1000

    trainsize = int(data_size*0.8)
    validsize = int(data_size*0.1)
    testsize = data_size - trainsize - validsize

    length = [trainsize, validsize, testsize]
    train_dataset, valid_dataset, test_dataset = random_split(dataset, length, generator=torch.Generator().manual_seed(35753))

    sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    sampler_valid = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=sampler_valid, shuffle=False)

    GAT_model = GAT(num_node_features=dataset.data.x.shape[1]).to(device)
    model = torch.nn.parallel.DistributedDataParallel(GAT_model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=False)
    print(model)


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0., betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-7, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=15,
    #     verbose=True,
    #     min_lr=1e-6
    # )

    criterion = LabelSmoothMSELoss(smoothing=0.001)

    for epoch in range(epochs):
        model.train()
        loss_sum_train = 0
        list_of_trloss = [torch.zeros(1).to(device) for _ in range(4)]
        train_loader.sampler.set_epoch(epoch)
        for data in train_loader:
            optimizer.zero_grad()
            # data = graph_data_augmentation(data)
            pred_tc = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(pred_tc, data.y)
            loss_sum_train = loss_sum_train + loss
            # 添加L1正则化
            # l1_lambda = 5e-6
            # l1_norm = sum(p.abs().sum() for p in model.parameters())
            # loss = loss + l1_lambda * l1_norm
            #
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            loss.backward()
            optimizer.step()
        avg_train_loss = loss_sum_train/len(train_loader)
        writer.add_scalar('train_loss', avg_train_loss, epoch)
        torch.cuda.synchronize()
        dist.all_gather(list_of_trloss, avg_train_loss)  # 收集所有进程的输出到list_of_outputs中
        average_trloss = torch.mean(torch.stack(list_of_trloss), dim=0)
        writer.add_scalar('train_loss_average', average_trloss, epoch)

        loss_sum_test = 0
        model.eval()
        list_of_loss = [torch.zeros(1).to(device) for _ in range(4)]
        for data in valid_loader:
            test_tc = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(test_tc, data.y)
            loss_sum_test = loss_sum_test + loss

        avg_test_loss = torch.tensor(loss_sum_test/len(valid_loader)).to(device)
        writer.add_scalar('test_loss', avg_test_loss, epoch)
        comm.synchronize()

        scheduler.step()
        dist.all_gather(list_of_loss, avg_test_loss)  # 收集所有进程的输出到list_of_outputs中
        average_loss = torch.mean(torch.stack(list_of_loss), dim=0)
        writer.add_scalar('test_loss_average', average_loss, epoch)

        if average_loss.item() <= best_loss and comm.is_main_process():
            torch.save(model.module.state_dict(), 'best_gtc_gat.pth')
            best_loss = average_loss.item()

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('lr', lr, epoch)


        print(f'RANK: {local_rank}, Epoch: {epoch:03d}, Train loss: {avg_train_loss:.4f}({average_trloss.item():.4f}), Test loss: {avg_test_loss:.4f}({average_loss.item():.4f}), lr: {lr}')
        if comm.is_main_process():
            torch.save(model.module.state_dict(), 'last_gtc_gat.pth')


    writer.close()


if __name__ == "__main__":
    main()




