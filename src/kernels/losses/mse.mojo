from gpu import thread_idx, block_idx, grid_dim
from layout import Layout, LayoutTensor

fn mse_forward[
    tpb: Int,
    dtype: DType,
    output_layout: Layout,
    predictions_layout: Layout,
    targets_layout: Layout
](
    output: LayoutTensor[dtype, output_layout, MutAnyOrigin],
    predictions: LayoutTensor[dtype, predictions_layout, ImmutAnyOrigin],
    targets: LayoutTensor[dtype, targets_layout, ImmutAnyOrigin],
    batch_size: Int,
    size1: Int,
    size2: Int
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    
    blocks_per_batch = ((size2 + tpb - 1) // tpb) * ((size1 + tpb - 1) // tpb)
    block_id = block_idx.x + block_idx.y * grid_dim.x + block_idx.z * grid_dim.x * grid_dim.y
    batch_idx = block_id // blocks_per_batch
    batch_block_id = block_id % blocks_per_batch
    if batch_idx >= batch_size:
        return
    
    blocks_x = (size2 + tpb - 1) // tpb
    block_row = batch_block_id // blocks_x
    block_col = batch_block_id % blocks_x
    
    global_row = block_row * tpb + local_row
    global_col = block_col * tpb + local_col
    
    if global_row < size1 and global_col < size2:
        pred = predictions[batch_idx, global_row, global_col]
        target = targets[batch_idx, global_row, global_col]
        diff = pred - target
        output[batch_idx, global_row, global_col] = diff * diff


fn mse_backward[
    tpb: Int,
    dtype: DType,
    output_input_gradient_layout: Layout,
    predictions_layout: Layout,
    targets_layout: Layout,
    local_gradient_layout: Layout
](
    output_input_gradient: LayoutTensor[dtype, output_input_gradient_layout, MutAnyOrigin],
    predictions: LayoutTensor[dtype, predictions_layout, ImmutAnyOrigin],
    targets: LayoutTensor[dtype, targets_layout, ImmutAnyOrigin],
    local_gradient: LayoutTensor[dtype, local_gradient_layout, ImmutAnyOrigin],
    batch_size: Int,
    size1: Int,
    size2: Int
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    
    blocks_per_batch = ((size2 + tpb - 1) // tpb) * ((size1 + tpb - 1) // tpb)
    block_id = block_idx.x + block_idx.y * grid_dim.x + block_idx.z * grid_dim.x * grid_dim.y
    batch_idx = block_id // blocks_per_batch
    batch_block_id = block_id % blocks_per_batch
    if batch_idx >= batch_size:
        return
    
    blocks_x = (size2 + tpb - 1) // tpb
    block_row = batch_block_id // blocks_x
    block_col = batch_block_id % blocks_x
    
    global_row = block_row * tpb + local_row
    global_col = block_col * tpb + local_col
    
    if global_row < size1 and global_col < size2:
        pred = predictions[batch_idx, global_row, global_col]
        target = targets[batch_idx, global_row, global_col]
        grad = local_gradient[batch_idx, global_row, global_col]
        output_input_gradient[batch_idx, global_row, global_col] = 2 * (pred - target) * grad
