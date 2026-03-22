import torch
import numpy as np
import math
from .data_structures import GraphData
from .config import Config


def restructure(graph: GraphData, config: Config) -> GraphData:
    device = graph.device
    graph.to("cpu")

    N = graph.num_neurons
    E = graph.num_edges

    sources_np = graph.sources.numpy()
    dests_np = graph.dests.numpy()
    weights_np = graph.weights.numpy()
    neuron_type_np = graph.neuron_type.numpy()
    grad_mean_np = graph.gradient_running_mean.numpy()
    mean_act_np = graph.mean_activation_count.numpy()
    has_cloned_np = graph.has_cloned.numpy()

    weight_mean_np = np.zeros(N, dtype=np.float32)
    out_counts = np.zeros(N, dtype=np.int32)
    for i in range(E):
        s = sources_np[i]
        weight_mean_np[s] += abs(weights_np[i])
        out_counts[s] += 1
    for i in range(N):
        if out_counts[i] > 0:
            weight_mean_np[i] /= out_counts[i]

    hidden_mask = neuron_type_np == 1
    hidden_indices = np.where(hidden_mask)[0]

    to_delete = set()
    to_clone = []

    for idx in hidden_indices:
        if mean_act_np[idx] < config.low_activation_threshold:
            to_delete.add(int(idx))
        elif grad_mean_np[idx] < config.low_gradient_threshold \
                and weight_mean_np[idx] > config.high_weight_threshold \
                and has_cloned_np[idx] == 0:
            to_clone.append(int(idx))

    active_hidden = [idx for idx in hidden_indices if int(idx) not in to_delete]
    if len(active_hidden) > 1:
        hidden_grads = np.array([grad_mean_np[idx] for idx in active_hidden])
        sorted_order = np.argsort(hidden_grads)
        n = min(config.restructure_top_n, len(active_hidden) // 2)
        if n > 0:
            low_grad_neurons = [int(active_hidden[sorted_order[i]]) for i in range(n)]
            high_grad_neurons = [int(active_hidden[sorted_order[-(i+1)]]) for i in range(n)]
            for donor in low_grad_neurons:
                for target in high_grad_neurons:
                    sources_np = np.append(sources_np, donor)
                    dests_np = np.append(dests_np, target)
                    weights_np = np.append(weights_np, np.float32(np.random.normal(0, 0.01)))

    new_neurons_defaults = []
    new_neurons_type = []
    new_edges_src = []
    new_edges_dst = []
    new_edges_w = []

    next_neuron_id = N

    max_clones = max(1, len(hidden_indices) // 10)
    if len(to_clone) > max_clones:
        to_clone = list(np.random.choice(to_clone, max_clones, replace=False))

    for clone_src in to_clone:
        clone_id = next_neuron_id
        next_neuron_id += 1
        has_cloned_np[clone_src] = 1

        new_neurons_defaults.append(graph.defaults[clone_src].item())
        new_neurons_type.append(1)

        in_edges = np.where(dests_np == clone_src)[0]
        for e in in_edges:
            new_edges_src.append(int(sources_np[e]))
            new_edges_dst.append(clone_id)
            new_edges_w.append(float(weights_np[e]) + np.random.normal(0, 0.01))

        out_edges = np.where(sources_np == clone_src)[0]
        for e in out_edges:
            new_edges_src.append(clone_id)
            new_edges_dst.append(int(dests_np[e]))
            new_edges_w.append(float(weights_np[e]) + np.random.normal(0, 0.01))

    keep_edges_mask = np.ones(len(sources_np), dtype=bool)
    for e_idx in range(len(sources_np)):
        if int(sources_np[e_idx]) in to_delete or int(dests_np[e_idx]) in to_delete:
            keep_edges_mask[e_idx] = False

    sources_np = sources_np[keep_edges_mask]
    dests_np = dests_np[keep_edges_mask]
    weights_np = weights_np[keep_edges_mask]

    if new_edges_src:
        sources_np = np.concatenate([sources_np, np.array(new_edges_src, dtype=np.int32)])
        dests_np = np.concatenate([dests_np, np.array(new_edges_dst, dtype=np.int32)])
        weights_np = np.concatenate([weights_np, np.array(new_edges_w, dtype=np.float32)])

    keep_neurons = sorted(set(range(N)) - to_delete)
    num_new_neurons = len(new_neurons_defaults)
    total_new_N = len(keep_neurons) + num_new_neurons

    old_to_new = {}
    for new_idx, old_idx in enumerate(keep_neurons):
        old_to_new[old_idx] = new_idx
    for i in range(num_new_neurons):
        old_to_new[N + i] = len(keep_neurons) + i

    new_sources = np.array([old_to_new[int(s)] for s in sources_np], dtype=np.int32)
    new_dests = np.array([old_to_new[int(d)] for d in dests_np], dtype=np.int32)

    new_graph = GraphData(device=device)
    new_graph.num_neurons = total_new_N
    new_graph.num_edges = len(new_sources)

    old_values = graph.values.numpy()
    old_defaults = graph.defaults.numpy()
    old_grad_mean = graph.gradient_running_mean.numpy()
    old_mean_act = graph.mean_activation_count.numpy()
    old_ntype = graph.neuron_type.numpy()

    new_values = np.zeros(total_new_N, dtype=np.float32)
    new_defaults = np.zeros(total_new_N, dtype=np.float32)
    new_grad_mean = np.zeros(total_new_N, dtype=np.float32)
    new_mean_act = np.zeros(total_new_N, dtype=np.float32)
    new_ntype = np.ones(total_new_N, dtype=np.int32)
    new_has_cloned = np.zeros(total_new_N, dtype=np.int32)

    for old_idx, new_idx in old_to_new.items():
        if old_idx < N:
            new_values[new_idx] = old_values[old_idx]
            new_defaults[new_idx] = old_defaults[old_idx]
            new_grad_mean[new_idx] = old_grad_mean[old_idx]
            new_mean_act[new_idx] = old_mean_act[old_idx]
            new_ntype[new_idx] = old_ntype[old_idx]
            new_has_cloned[new_idx] = has_cloned_np[old_idx]

    for i in range(num_new_neurons):
        new_idx = len(keep_neurons) + i
        new_defaults[new_idx] = new_neurons_defaults[i]
        new_ntype[new_idx] = new_neurons_type[i]

    new_graph.values = torch.from_numpy(new_values)
    new_graph.defaults = torch.from_numpy(new_defaults)
    new_graph.activation_counts = torch.zeros(total_new_N, dtype=torch.int32)
    new_graph.grad_accum = torch.zeros(total_new_N, dtype=torch.float32)
    new_graph.gradient_running_mean = torch.from_numpy(new_grad_mean)
    new_graph.weight_mean = torch.zeros(total_new_N, dtype=torch.float32)
    new_graph.mean_activation_count = torch.from_numpy(new_mean_act)
    new_graph.neuron_type = torch.from_numpy(new_ntype)
    new_graph.has_cloned = torch.from_numpy(new_has_cloned)

    new_graph.sources = torch.from_numpy(new_sources)
    new_graph.dests = torch.from_numpy(new_dests)
    new_graph.weights = torch.from_numpy(weights_np.copy())
    new_graph.grad_accum_w = torch.zeros(new_graph.num_edges, dtype=torch.float32)

    new_graph.input_indices = torch.where(new_graph.neuron_type == 0)[0]
    new_graph.output_indices = torch.where(new_graph.neuron_type == 2)[0]

    new_graph.rebuild_csr()
    new_graph.to(device)

    return new_graph
