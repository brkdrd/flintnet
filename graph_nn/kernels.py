import triton
import triton.language as tl
import torch


@triton.jit
def _activation(x, fn_id):
    if fn_id == 0:
        return tl.where(x > 0, x, 0.0)
    elif fn_id == 1:
        return tl.sigmoid(x)
    elif fn_id == 2:
        return tl.math.tanh(x)
    return x


@triton.jit
def _activation_deriv(output, fn_id):
    if fn_id == 0:
        return tl.where(output > 0, 1.0, 0.0)
    elif fn_id == 1:
        return output * (1.0 - output)
    elif fn_id == 2:
        return 1.0 - output * output
    return tl.full(output.shape, 1.0, dtype=output.dtype)


@triton.jit
def forward_kernel(
    batch_ptr,
    batch_size,
    values_ptr,
    defaults_ptr,
    activation_counts_ptr,
    weights_ptr,
    sources_ptr,
    in_offsets_ptr,
    in_edge_indices_ptr,
    out_offsets_ptr,
    out_edge_indices_ptr,
    dests_ptr,
    neuron_type_ptr,
    next_queue_ptr,
    queue_counter_ptr,
    completion_flag_ptr,
    queued_ptr,
    max_activations: tl.constexpr,
    activation_fn_id: tl.constexpr,
    activation_threshold: tl.constexpr,
    MAX_FAN_IN: tl.constexpr,
    MAX_FAN_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    neuron_idx = tl.load(batch_ptr + pid)

    # Compute weighted sum (no-op for inputs since fan_in=0)
    in_start = tl.load(in_offsets_ptr + neuron_idx)
    in_end = tl.load(in_offsets_ptr + neuron_idx + 1)
    fan_in = in_end - in_start

    acc = tl.load(defaults_ptr + neuron_idx)

    for i in range(MAX_FAN_IN):
        if i < fan_in:
            edge_idx = tl.load(in_edge_indices_ptr + in_start + i)
            src = tl.load(sources_ptr + edge_idx)
            w = tl.load(weights_ptr + edge_idx)
            v = tl.load(values_ptr + src)
            acc += w * v

    activated = _activation(acc, activation_fn_id)

    ntype = tl.load(neuron_type_ptr + neuron_idx)

    # Input neurons: use pre-set value instead of computed
    if ntype == 0:
        activated = tl.load(values_ptr + neuron_idx)
    # Output neurons: use raw logits (no activation)
    if ntype == 2:
        activated = acc

    # Store and update counts for non-input neurons
    if ntype != 0:
        tl.store(values_ptr + neuron_idx, activated)
        act_count = tl.load(activation_counts_ptr + neuron_idx)
        tl.store(activation_counts_ptr + neuron_idx, act_count + 1)

    if ntype == 2:
        tl.store(completion_flag_ptr, 1)

    if activated > activation_threshold:
        out_start = tl.load(out_offsets_ptr + neuron_idx)
        out_end = tl.load(out_offsets_ptr + neuron_idx + 1)
        fan_out = out_end - out_start

        for i in range(MAX_FAN_OUT):
            if i < fan_out:
                edge_idx = tl.load(out_edge_indices_ptr + out_start + i)
                dest = tl.load(dests_ptr + edge_idx)
                dest_act = tl.load(activation_counts_ptr + dest)
                if dest_act < max_activations:
                    was_queued = tl.atomic_add(queued_ptr + dest, 1)
                    if was_queued == 0:
                        slot = tl.atomic_add(queue_counter_ptr, 1)
                        tl.store(next_queue_ptr + slot, dest)


@triton.jit
def backward_kernel(
    batch_ptr,
    batch_size,
    values_ptr,
    activation_counts_ptr,
    weights_ptr,
    sources_ptr,
    dests_ptr,
    grad_accum_ptr,
    grad_accum_w_ptr,
    in_offsets_ptr,
    in_edge_indices_ptr,
    out_offsets_ptr,
    out_edge_indices_ptr,
    next_queue_ptr,
    queue_counter_ptr,
    visited_ptr,
    neuron_type_ptr,
    activation_fn_id: tl.constexpr,
    MAX_FAN_IN: tl.constexpr,
    MAX_FAN_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    neuron_idx = tl.load(batch_ptr + pid)

    act_count = tl.load(activation_counts_ptr + neuron_idx)
    if act_count <= 0:
        return

    neuron_val = tl.load(values_ptr + neuron_idx)
    neuron_grad = tl.load(grad_accum_ptr + neuron_idx)
    deriv = _activation_deriv(neuron_val, activation_fn_id)
    local_grad = neuron_grad * deriv
    ntype = tl.load(neuron_type_ptr + neuron_idx)
    # Output neurons have no activation, so derivative is 1.0
    if ntype == 2:
        local_grad = neuron_grad

    in_start = tl.load(in_offsets_ptr + neuron_idx)
    in_end = tl.load(in_offsets_ptr + neuron_idx + 1)
    fan_in = in_end - in_start

    for i in range(MAX_FAN_IN):
        if i < fan_in:
            edge_idx = tl.load(in_edge_indices_ptr + in_start + i)
            src = tl.load(sources_ptr + edge_idx)
            src_val = tl.load(values_ptr + src)
            tl.atomic_add(grad_accum_w_ptr + edge_idx, local_grad * src_val)
            tl.atomic_add(grad_accum_ptr + src, local_grad * tl.load(weights_ptr + edge_idx))

            src_act = tl.load(activation_counts_ptr + src)
            if src_act > 0:
                was_visited = tl.atomic_add(visited_ptr + src, 1)
                if was_visited == 0:
                    slot = tl.atomic_add(queue_counter_ptr, 1)
                    tl.store(next_queue_ptr + slot, src)

    tl.store(activation_counts_ptr + neuron_idx, act_count - 1)


@triton.jit
def apply_gradients_kernel(
    weights_ptr,
    defaults_ptr,
    grad_accum_w_ptr,
    grad_accum_ptr,
    num_edges,
    num_neurons,
    lr: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    edge_mask = offsets < num_edges
    w = tl.load(weights_ptr + offsets, mask=edge_mask, other=0.0)
    gw = tl.load(grad_accum_w_ptr + offsets, mask=edge_mask, other=0.0)
    tl.store(weights_ptr + offsets, w - lr * gw, mask=edge_mask)
    tl.store(grad_accum_w_ptr + offsets, 0.0, mask=edge_mask)

    neuron_mask = offsets < num_neurons
    d = tl.load(defaults_ptr + offsets, mask=neuron_mask, other=0.0)
    gd = tl.load(grad_accum_ptr + offsets, mask=neuron_mask, other=0.0)
    tl.store(defaults_ptr + offsets, d - lr * gd, mask=neuron_mask)
    tl.store(grad_accum_ptr + offsets, 0.0, mask=neuron_mask)


@triton.jit
def update_gradient_stats_kernel(
    grad_running_mean_ptr,
    grad_accum_ptr,
    out_offsets_ptr,
    out_edge_indices_ptr,
    dests_ptr,
    num_neurons,
    alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_neurons

    for i in range(BLOCK_SIZE):
        actual_idx = pid * BLOCK_SIZE + i
        if actual_idx < num_neurons:
            out_start = tl.load(out_offsets_ptr + actual_idx)
            out_end = tl.load(out_offsets_ptr + actual_idx + 1)
            fan_out = out_end - out_start

            if fan_out > 0:
                grad_sum = 0.0
                for j in range(1024):
                    if j < fan_out:
                        edge_idx = tl.load(out_edge_indices_ptr + out_start + j)
                        dest = tl.load(dests_ptr + edge_idx)
                        g = tl.load(grad_accum_ptr + dest)
                        grad_sum += tl.abs(g)
                avg_grad = grad_sum / fan_out.to(tl.float32)

                old_mean = tl.load(grad_running_mean_ptr + actual_idx)
                new_mean = (1.0 - alpha) * old_mean + alpha * avg_grad
                tl.store(grad_running_mean_ptr + actual_idx, new_mean)


def get_activation_fn_id(name: str) -> int:
    mapping = {"relu": 0, "sigmoid": 1, "tanh": 2}
    return mapping.get(name, 0)


def launch_forward_kernel(
    batch_indices, graph, config, next_queue, queue_counter, completion_flag, queued
):
    batch_size = batch_indices.shape[0]
    if batch_size == 0:
        return

    max_fan_in = max(int((graph.in_offsets[1:] - graph.in_offsets[:-1]).max().item()), 1)
    max_fan_out = max(int((graph.out_offsets[1:] - graph.out_offsets[:-1]).max().item()), 1)

    max_fan_in = triton.next_power_of_2(max_fan_in)
    max_fan_out = triton.next_power_of_2(max_fan_out)

    grid = (batch_size,)
    forward_kernel[grid](
        batch_indices,
        batch_size,
        graph.values,
        graph.defaults,
        graph.activation_counts,
        graph.weights,
        graph.sources,
        graph.in_offsets,
        graph.in_edge_indices,
        graph.out_offsets,
        graph.out_edge_indices,
        graph.dests,
        graph.neuron_type,
        next_queue,
        queue_counter,
        completion_flag,
        queued,
        max_activations=config.max_activations_per_neuron,
        activation_fn_id=get_activation_fn_id(config.activation_fn),
        activation_threshold=config.activation_threshold,
        MAX_FAN_IN=max_fan_in,
        MAX_FAN_OUT=max_fan_out,
    )


def launch_backward_kernel(
    batch_indices, graph, config, next_queue, queue_counter, visited
):
    batch_size = batch_indices.shape[0]
    if batch_size == 0:
        return

    max_fan_in = max(int((graph.in_offsets[1:] - graph.in_offsets[:-1]).max().item()), 1)
    max_fan_out = max(int((graph.out_offsets[1:] - graph.out_offsets[:-1]).max().item()), 1)

    max_fan_in = triton.next_power_of_2(max_fan_in)
    max_fan_out = triton.next_power_of_2(max_fan_out)

    grid = (batch_size,)
    backward_kernel[grid](
        batch_indices,
        batch_size,
        graph.values,
        graph.activation_counts,
        graph.weights,
        graph.sources,
        graph.dests,
        graph.grad_accum,
        graph.grad_accum_w,
        graph.in_offsets,
        graph.in_edge_indices,
        graph.out_offsets,
        graph.out_edge_indices,
        next_queue,
        queue_counter,
        visited,
        graph.neuron_type,
        activation_fn_id=get_activation_fn_id(config.activation_fn),
        MAX_FAN_IN=max_fan_in,
        MAX_FAN_OUT=max_fan_out,
    )


def launch_apply_gradients(graph, config):
    n = max(graph.num_edges, graph.num_neurons)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    apply_gradients_kernel[grid](
        graph.weights,
        graph.defaults,
        graph.grad_accum_w,
        graph.grad_accum,
        graph.num_edges,
        graph.num_neurons,
        lr=config.learning_rate,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def launch_update_gradient_stats(graph, config):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(graph.num_neurons, BLOCK_SIZE),)
    update_gradient_stats_kernel[grid](
        graph.gradient_running_mean,
        graph.grad_accum,
        graph.out_offsets,
        graph.out_edge_indices,
        graph.dests,
        graph.num_neurons,
        alpha=config.gradient_ema_alpha,
        BLOCK_SIZE=BLOCK_SIZE,
    )
