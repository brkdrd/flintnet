import torch
from .config import Config
from .data_structures import GraphData
from .init_topology import build_layered_graph
from .kernels import (
    launch_forward_kernel,
    launch_backward_kernel,
    launch_apply_gradients,
    launch_update_gradient_stats,
)
from .restructure import restructure


class Network:
    def __init__(self, layer_sizes: list[int], config: Config = None, device="cuda"):
        self.config = config or Config()
        self.device = device
        self.graph = build_layered_graph(layer_sizes, device=device)
        self.pass_counter = 0
        self.max_queue_size = self.graph.num_edges * 2

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        graph = self.graph
        config = self.config

        graph.values.zero_()
        graph.activation_counts.zero_()

        input_indices = graph.input_indices
        graph.values[input_indices] = input_data.to(self.device)
        graph.activation_counts[input_indices] = 1

        completion_flag = torch.zeros(1, dtype=torch.int32, device=self.device)

        queue = graph.input_indices.clone().to(torch.int32)
        next_queue = torch.zeros(self.max_queue_size, dtype=torch.int32, device=self.device)
        queue_counter = torch.zeros(1, dtype=torch.int32, device=self.device)

        iteration = 0
        max_iterations = config.max_activations_per_neuron * 2

        while iteration < max_iterations:
            if completion_flag.item() == 1:
                break

            batch_size = queue.shape[0]
            if batch_size == 0:
                break

            queue_counter.zero_()

            launch_forward_kernel(
                queue, graph, config, next_queue, queue_counter, completion_flag
            )

            torch.cuda.synchronize()

            count = queue_counter.item()
            if count == 0:
                break

            queue = next_queue[:count].clone()
            next_queue.zero_()
            iteration += 1

        output_values = graph.values[graph.output_indices].clone()
        return output_values

    def backward(self, targets: torch.Tensor):
        graph = self.graph
        config = self.config

        graph.grad_accum.zero_()
        graph.grad_accum_w.zero_()

        output_values = graph.values[graph.output_indices]
        if config.loss_fn == "mse":
            targets_dev = targets.to(self.device)
            grad = 2.0 * (output_values - targets_dev) / targets_dev.shape[0]
        elif config.loss_fn == "cross_entropy":
            targets_dev = targets.to(self.device)
            probs = torch.softmax(output_values, dim=0)
            grad = probs - targets_dev
        else:
            targets_dev = targets.to(self.device)
            grad = 2.0 * (output_values - targets_dev) / targets_dev.shape[0]

        graph.grad_accum[graph.output_indices] = grad

        visited = torch.zeros(graph.num_neurons, dtype=torch.int32, device=self.device)

        queue = graph.output_indices.clone().to(torch.int32)
        visited[graph.output_indices] = 1

        next_queue = torch.zeros(self.max_queue_size, dtype=torch.int32, device=self.device)
        queue_counter = torch.zeros(1, dtype=torch.int32, device=self.device)

        max_iterations = config.max_activations_per_neuron * 2

        for _ in range(max_iterations):
            batch_size = queue.shape[0]
            if batch_size == 0:
                break

            queue_counter.zero_()

            launch_backward_kernel(
                queue, graph, config, next_queue, queue_counter, visited
            )

            torch.cuda.synchronize()

            count = queue_counter.item()
            if count == 0:
                break

            queue = next_queue[:count].clone()
            next_queue.zero_()

        launch_update_gradient_stats(graph, config)
        launch_apply_gradients(graph, config)

        act_counts = graph.activation_counts.float()
        alpha = config.gradient_ema_alpha
        graph.mean_activation_count = (
            (1.0 - alpha) * graph.mean_activation_count + alpha * act_counts
        )

        self.pass_counter += 1
        if self.pass_counter % config.restructure_interval == 0:
            self.graph = restructure(self.graph, config)
            self.max_queue_size = self.graph.num_edges * 2

    def compute_loss(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_dev = targets.to(self.device)
        if self.config.loss_fn == "mse":
            return torch.mean((output - targets_dev) ** 2)
        elif self.config.loss_fn == "cross_entropy":
            log_probs = torch.log_softmax(output, dim=0)
            return -torch.sum(targets_dev * log_probs)
        return torch.mean((output - targets_dev) ** 2)
