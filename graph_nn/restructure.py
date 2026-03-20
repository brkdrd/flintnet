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
    high_err_high_w = []
    low_err_low_w = []
    to_clone = []

    for idx in hidden_indices:
        high_grad = grad_mean_np[idx] > config.high_gradient_threshold
        low_grad = grad_mean_np[idx] < config.low_gradient_threshold
        high_w = weight_mean_np[idx] > config.high_weight_threshold
        low_w = weight_mean_np[idx] < config.low_weight_threshold

        if high_grad and low_w:
            to_delete.add(int(idx))
        elif high_grad and high_w:
            high_err_high_w.append(int(idx))
        elif low_grad and low_w:
            low_err_low_w.append(int(idx))
        elif low_grad and high_w:
            to_clone.append(int(idx))

        if mean_act_np[idx] < config.low_activation_threshold:
            to_delete.add(int(idx))

    for h_neuron in high_err_high_w:
        for donor in low_err_low_w:
            if donor not in to_delete:
                sources_np = np.append(sources_np, donor)
                dests_np = np.append(dests_np, h_neuron)
                weights_np = np.append(weights_np, np.float32(np.random.normal(0, 0.01)))

    new_neurons_defaults = []
    new_neurons_type = []
    new_edges_src = []
    new_edges_dst = []
    new_edges_w = []

    next_neuron_id = N

    for clone_src in to_clone:
        clone_id = next_neuron_id
        next_neuron_id += 1

        new_neurons_defaults.append(graph.defaults[clone_src].item())
        new_neurons_type.append(1)

        in_edges = np.where(dests_np == clone_src)[0]
        low_err_set = set(low_err_low_w) | set(to_clone)
        eligible_in = [e for e in in_edges if int(sources_np[e]) in low_err_set]

        np.random.shuffle(eligible_in)
        split = len(eligible_in) // 2
        for e in eligible_in[:split]:
            new_edges_src.append(int(sources_np[e]))
            new_edges_dst.append(clone_id)
            new_edges_w.append(float(weights_np[e]))

        out_edges = np.where(sources_np == clone_src)[0]
        high_w_set = set(high_err_high_w) | set(to_clone)
        eligible_out = [e for e in out_edges if int(dests_np[e]) in high_w_set]

        np.random.shuffle(eligible_out)
        split_out = len(eligible_out) // 2
        for e in eligible_out[:split_out]:
            new_edges_src.append(clone_id)
            new_edges_dst.append(int(dests_np[e]))
            new_edges_w.append(float(weights_np[e]))

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

    for old_idx, new_idx in old_to_new.items():
        if old_idx < N:
            new_values[new_idx] = old_values[old_idx]
            new_defaults[new_idx] = old_defaults[old_idx]
            new_grad_mean[new_idx] = old_grad_mean[old_idx]
            new_mean_act[new_idx] = old_mean_act[old_idx]
            new_ntype[new_idx] = old_ntype[old_idx]

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

    new_graph.sources = torch.from_numpy(new_sources)
    new_graph.dests = torch.from_numpy(new_dests)
    new_graph.weights = torch.from_numpy(weights_np.copy())
    new_graph.grad_accum_w = torch.zeros(new_graph.num_edges, dtype=torch.float32)

    new_graph.input_indices = torch.where(new_graph.neuron_type == 0)[0]
    new_graph.output_indices = torch.where(new_graph.neuron_type == 2)[0]

    new_graph.rebuild_csr()
    new_graph.to(device)

    return new_graph
