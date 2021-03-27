#!/usr/bin/env python3
import torch
import numpy as np
import time
class TCGNN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, dim, num_class):
        super(TCGNN_dataset, self).__init__()
        self.nodes = set()
        self.num_nodes = 0
        self.num_features = dim 
        self.num_classes = num_class
        self.init_edges(path)
        self.init_embedding(dim)
        self.init_labels(num_class)
        train = 1
        val = 0.3
        test = 0.1
        self.train_mask = [1] * int(self.num_nodes * train) + [0] * (self.num_nodes  - int(self.num_nodes * train))
        self.val_mask = [1] * int(self.num_nodes * val)+ [0] * (self.num_nodes  - int(self.num_nodes * val))
        self.test_mask = [1] * int(self.num_nodes * test) + [0] * (self.num_nodes  - int(self.num_nodes * test))
        self.train_mask = torch.BoolTensor(self.train_mask).cuda()
        self.val_mask = torch.BoolTensor(self.val_mask).cuda()
        self.test_mask = torch.BoolTensor(self.test_mask).cuda()

    def init_edges(self, path):
        fp = open(path, "r")
        src_li = []
        dst_li = []
        start = time.perf_counter()
        for line in fp:
            src, dst = line.strip('\n').split()
            src, dst = int(src), int(dst)
            src_li.append(src)
            dst_li.append(dst)
            self.nodes.add(src)
            self.nodes.add(dst)
        self.edge_index = np.array([src_li, dst_li])
        self.num_nodes = max(self.nodes) + 1
        dur = time.perf_counter() - start
        print("Loading (ms):\t{:.3f}".format(dur*1e3))


    def init_embedding(self, dim):
        self.x = torch.randn(self.num_nodes, dim).cuda()
    
    def init_labels(self, num_class):
        self.y = torch.LongTensor([1] * self.num_nodes).cuda()

    def forward(*input, **kwargs):
        pass
