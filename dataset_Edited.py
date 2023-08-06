#!/usr/bin/env python3
import torch
import numpy as np
import time
import pubmed_util
import create_graph_dgl as cg

from config import *
from scipy.sparse import *

class TCGNN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, num_classes, verbose = False):
        super(TCGNN_dataset, self).__init__()

        self.num_classes = num_classes

        self.feature = pubmed_util.read_feature_info(path + "/feature/feature.txt")

# check which one to check accuracy for, plus also which to load
        self.train_id = pubmed_util.read_index_info(path + "/index/train_index.txt") #480
        self.test_id = pubmed_util.read_index_info(path + "/index/test_index.txt") #1000

        self.train_y_label =  pubmed_util.read_label_info(path + "/label/y_label.txt") #
        self.test_y_label =  pubmed_util.read_label_info(path + "/label/test_y_label.txt") #1000

        self.verbose_flag = verbose

        graph_data_path = path + "/graph_structure/graph.txt"
        line_needed_to_skip = 0
        self.src_li, self.dst_li, comment = cg.read_graph_data(graph_data_path, line_needed_to_skip)

        self.num_nodes = len(self.feature) # might have to check if it is correct, can use 
                                           # cg.build_graph() for graph g, then g.get_num_nodes()
        self.num_edges = len(self.src_li)
        self.edge_index = np.stack([self.src_li, self.dst_li])

        #self.train_y_label, test_y_label, train_id, test_id =  pubmed_util.ran_init_index_and_label(args.category, num_train, num_test)
        #acc_val2 = pubmed_util.accuracy(logp[test_id],test_y_label)

        self.avg_degree = self.num_edges / self.num_nodes
        self.avg_edgeSpan = np.mean(np.abs(np.subtract(self.src_li, self.dst_li)))
    
        # Build graph CSR.

        val = [1] * self.num_edges
        start = time.perf_counter()
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        scipy_csr = scipy_coo.tocsr()
        build_csr = time.perf_counter() - start

        if self.verbose_flag:
            print("# Build CSR (s): {:.3f}".format(build_csr))

        self.column_index = torch.IntTensor(scipy_csr.indices)
        self.row_pointers = torch.IntTensor(scipy_csr.indptr)

        # Get degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()

        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()

        self.x = torch.tensor(self.feature).cuda()
        #self.x = torch.tensor(self.feature)[self.train_id].cuda()
        self.y = torch.tensor(self.train_y_label).cuda()
        
        self.num_features = self.x.shape[1]

        print('num nodes: ', self.num_nodes,' - ', self.x.shape[0])
        print('num features: ', self.num_features)
        print('x shape: ', self.x.shape)
        print('y shape: ', self.y.shape)
        #remove it later
        print(path)

    def init_edges(self, path):
        None
        # loading from a txt graph file
        '''
        if self.load_from_txt:
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
            
            self.num_edges = len(src_li)
            self.num_nodes = max(self.nodes) + 1
            self.edge_index = np.stack([src_li, dst_li])

            dur = time.perf_counter() - start
            if self.verbose_flag:
                print("# Loading (txt) {:.3f}s ".format(dur))
                
        

        # loading from a .npz graph file
        else: 
            
            if not path.endswith('.npz'):
                raise ValueError("graph file must be a .npz file")

            start = time.perf_counter()
            graph_obj = np.load(path)
            src_li = graph_obj['src_li']
            dst_li = graph_obj['dst_li']

            self.num_nodes = graph_obj['num_nodes']
            self.num_edges = len(src_li)
            self.edge_index = np.stack([src_li, dst_li])
            dur = time.perf_counter() - start
            if self.verbose_flag:
                print("# Loading (npz)(s): {:.3f}".format(dur))
        
        self.avg_degree = self.num_edges / self.num_nodes
        self.avg_edgeSpan = np.mean(np.abs(np.subtract(src_li, dst_li)))

        if self.verbose_flag:
            print('# nodes: {}'.format(self.num_nodes))
            print("# avg_degree: {:.2f}".format(self.avg_degree))
            print("# avg_edgeSpan: {}".format(int(self.avg_edgeSpan)))

        # Build graph CSR.
        val = [1] * self.num_edges
        start = time.perf_counter()
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        scipy_csr = scipy_coo.tocsr()
        build_csr = time.perf_counter() - start

        if self.verbose_flag:
            print("# Build CSR (s): {:.3f}".format(build_csr))

        self.column_index = torch.IntTensor(scipy_csr.indices)
        self.row_pointers = torch.IntTensor(scipy_csr.indptr)

        # Get degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        '''

    def init_embedding(self, dim):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        
        self.x = torch.randn(self.num_nodes, dim).cuda()
        '''
        None
    
    def init_labels(self, num_class):
        '''
        Generate the node label.
        Called from __init__.
        
        self.y = torch.ones(self.num_nodes).long().cuda()
        '''
        None
