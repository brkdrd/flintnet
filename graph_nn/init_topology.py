import torch
import numpy as np
import math
from .data_structures import GraphData


def build_layered_graph(layer_sizes: list[int], device="cuda") -> GraphData:
    graph = GraphData(device=device)

    N = sum(layer_sizes)
    graph.num_neurons = N

    graph.values = torch.zeros(N, dtype=torch.float32, device=device)
    graph.defaults = torch.zeros(N, dtype=torch.float32, device=device)
    graph.activation_counts = torch.zeros(N, dtype=torch.int32, device=device)
    graph.grad_accum = torch.zeros(N, dtype=torch.float32, device=device)
    graph.gradient_running_mean = torch.zeros(N, dtype=torch.float32, device=device)
    graph.weight_mean = torch.zeros(N, dtype=torch.float32, device=device)
    graph.mean_activation_count = torch.zeros(N, dtype=torch.float32, device=device)

    neuron_type = torch.ones(N, dtype=torch.int32, device=device)
    offset = 0
    layer_offsets = []
    for i, size in enumerate(layer_sizes):
        layer_offsets.append(offset)
        if i == 0:
            neuron_type[offset:offset + size] = 0
        elif i == len(layer_sizes) - 1:
            neuron_type[offset:offset + size] = 2
        offset += size
    graph.neuron_type = neuron_type

    graph.has_cloned = torch.zeros(N, dtype=torch.int32, device=device)
    graph.input_indices = torch.where(neuron_type == 0)[0].to(device)
    graph.output_indices = torch.where(neuron_type == 2)[0].to(device)

    edges_src = []
    edges_dst = []
    edge_weights = []

    for layer_idx in range(len(layer_sizes) - 1):
        src_start = layer_offsets[layer_idx]
        src_size = layer_sizes[layer_idx]
        dst_start = layer_offsets[layer_idx + 1]
        dst_size = layer_sizes[layer_idx + 1]

        fan_in = src_size
        std = math.sqrt(2.0 / fan_in)

        for s in range(src_size):
            for d in range(dst_size):
                edges_src.append(src_start + s)
                edges_dst.append(dst_start + d)
                edge_weights.append(np.random.normal(0, std))

    E = len(edges_src)
    graph.num_edges = E

    graph.sources = torch.tensor(edges_src, dtype=torch.int32, device=device)
    graph.dests = torch.tensor(edges_dst, dtype=torch.int32, device=device)
    graph.weights = torch.tensor(edge_weights, dtype=torch.float32, device=device)
    graph.grad_accum_w = torch.zeros(E, dtype=torch.float32, device=device)

    graph.rebuild_csr()

    return graph
