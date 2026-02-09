from gpu import thread_idx, block_idx, grid_dim
from layout import Layout, LayoutTensor

fn relu_forward[
    tpb: Int,
    dtype: DType,
    output_layout: Layout,
    x_layout: Layout
](
    output: LayoutTensor[dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[dtype, x_layout, ImmutAnyOrigin],
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
        val = x[batch_idx, global_row, global_col]
        output[batch_idx, global_row, global_col] = max(val, 0)

fn relu_backward[
    tpb: Int,
    dtype: DType,
    x_grad_layout: Layout,
    x_layout: Layout,
    local_gradient_layout: Layout
](
    x_gradient: LayoutTensor[dtype, x_grad_layout, MutAnyOrigin],
    x: LayoutTensor[dtype, x_layout, ImmutAnyOrigin],
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
        x_val = x[batch_idx, global_row, global_col]
        grad = local_gradient[batch_idx, global_row, global_col]
        x_gradient[batch_idx, global_row, global_col] = grad if x_val > 0 else 0
