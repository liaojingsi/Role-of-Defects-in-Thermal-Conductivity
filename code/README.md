# Deep Learning Models for Thermal Conductivity Prediction

## Directory Structure

```
code/
├── ANN/                  # MLP with RDF features
│   ├── RDF_model.py      # Training script
│   ├── RDF_test.py       # Testing script
│   ├── best_gtc_gat.pth  # Best model weights
│   └── tc_test_GAT.png   # Result figure
│
├── CNN/                  # 1D CNN with RDF features
│   ├── CNNRDF_model.py
│   ├── CNNRDF_test.py
│   ├── best_gtc_gat.pth
│   └── tc_test_GAT.png
│
├── AEANN/                # Residual attention MLP with RDF features
│   ├── ATRDF_model.py
│   ├── ATRDF_test.py
│   ├── best_gtc_gat.pth
│   └── tc_test_GAT.png
│
├── GCN/                  # Graph convolutional network
│   ├── GCN_TC_model.py
│   ├── GCN_TC_test.py
│   ├── GTC.py            # PyG dataset utility
│   ├── best_gtc_gat.pth
│   └── tc_test_GAT.png
│
├── GAT/                  # Graph attention network
│   ├── GAT.py
│   ├── GAT_test.py
│   ├── GTC.py            # PyG dataset utility
│   ├── best_gtc_gat.pth
│   └── tc_test_GAT.png
│
├── GAT-RDF/              # Graph attention network with RDF-fused dataset
│   ├── GAT.py
│   ├── GAT_test.py
│   ├── GTC.py            # PyG dataset utility
│   ├── best_gtc_gat.pth
│   └── tc_test_GAT.png
│
├── toy_dataset.rar       # Example dataset
└── README.md
```

## Notes

1. Dataset root paths in the scripts (e.g. `/data/jsliao/GNN/GTC_n18`) are training-server paths; modify them to your local paths before running.
2. Training scripts use PyTorch DistributedDataParallel and depend on a custom utility module `util.comm`; launch with `torchrun --nproc_per_node=N <train_script>.py`.
3. The `GTC.py` in GCN/GAT/GAT-RDF is a custom `torch_geometric.InMemoryDataset` class for loading the TU-format graph data.
4. Test scripts load `best_gtc_gat.pth` and require a GPU; change the device to `cpu` if unavailable.
5. `toy_dataset.rar` is a small example dataset for verifying the pipeline; extract it before use.
