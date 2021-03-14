"""By convention, masks should have true elements at positions where higher precision should be used"""

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import degree


class ProbabilisticHighDegreeMask:
    def __init__(self, low_quantise_prob, high_quantise_prob, per_graph=True):
        self.low_prob = low_quantise_prob
        self.high_prob = high_quantise_prob
        self.per_graph = per_graph

    def _process_graph(self, graph):
        # Note that:
        # 1. The probability of being protected increases as the indegree increases
        # 2. All nodes with the same indegree have the same bernoulli p
        # 3. you can set this such that all nodes have some probability of being quantised

        n = graph.num_nodes
        indegree = degree(graph.edge_index[1], n, dtype=torch.long)
        counts = torch.bincount(indegree)

        step_size = (self.high_prob - self.low_prob) / n
        indegree_ps = counts * step_size
        indegree_ps = torch.cumsum(indegree_ps, dim=0)
        indegree_ps += self.low_prob
        graph.prob_mask = indegree_ps[indegree]

        return graph

    def __call__(self, data):
        if self.per_graph and isinstance(data, Batch):
            graphs = data.to_data_list()
            processed = []
            for g in graphs:
                g = self._process_graph(g)
                processed.append(g)
            return Batch.from_data_list(processed)
        else:
            return self._process_graph(data)
