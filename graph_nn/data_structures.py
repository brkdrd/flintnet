import torch
import numpy as np


class GraphData:
    def __init__(self, device="cuda"):
        self.device = device

        self.num_neurons = 0
        self.num_edges = 0

        self.values = None
        self.defaults = None
        self.activation_counts = None
        self.grad_accum = None
        self.gradient_running_mean = None
        self.weight_mean = None
        self.mean_activation_count = None
        self.neuron_type = None

        self.weights = None
        self.sources = None
        self.dests = None
        self.grad_accum_w = None

        self.in_offsets = None
        self.in_edge_indices = None
        self.out_offsets = None
        self.out_edge_indices = None

        self.has_cloned = None

        self.input_indices = None
        self.output_indices = None

    def rebuild_csr(self):
        N = self.num_neurons
        E = self.num_edges

        sources_cpu = self.sources.cpu().numpy()
        dests_cpu = self.dests.cpu().numpy()

        in_counts = np.zeros(N, dtype=np.int32)
        out_counts = np.zeros(N, dtype=np.int32)
        for i in range(E):
            in_counts[dests_cpu[i]] += 1
            out_counts[sources_cpu[i]] += 1

        in_offsets = np.zeros(N + 1, dtype=np.int32)
        out_offsets = np.zeros(N + 1, dtype=np.int32)
        for i in range(N):
            in_offsets[i + 1] = in_offsets[i] + in_counts[i]
            out_offsets[i + 1] = out_offsets[i] + out_counts[i]

        in_edge_indices = np.zeros(E, dtype=np.int32)
        out_edge_indices = np.zeros(E, dtype=np.int32)
        in_fill = np.zeros(N, dtype=np.int32)
        out_fill = np.zeros(N, dtype=np.int32)

        for edge_idx in range(E):
            s = sources_cpu[edge_idx]
            d = dests_cpu[edge_idx]

            pos_in = in_offsets[d] + in_fill[d]
            in_edge_indices[pos_in] = edge_idx
            in_fill[d] += 1

            pos_out = out_offsets[s] + out_fill[s]
            out_edge_indices[pos_out] = edge_idx
            out_fill[s] += 1

        self.in_offsets = torch.from_numpy(in_offsets).to(self.device)
        self.in_edge_indices = torch.from_numpy(in_edge_indices).to(self.device)
        self.out_offsets = torch.from_numpy(out_offsets).to(self.device)
        self.out_edge_indices = torch.from_numpy(out_edge_indices).to(self.device)

    def to(self, device):
        self.device = device
        for attr in [
            "values", "defaults", "activation_counts", "grad_accum",
            "gradient_running_mean", "weight_mean", "mean_activation_count",
            "neuron_type", "weights", "sources", "dests", "grad_accum_w",
            "has_cloned",
            "in_offsets", "in_edge_indices", "out_offsets", "out_edge_indices",
            "input_indices", "output_indices",
        ]:
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))
        return self
