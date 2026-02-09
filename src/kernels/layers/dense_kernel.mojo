from gpu import thread_idx, block_idx, grid_dim, barrier
from layout import Layout, LayoutTensor

fn dense_forward[
    tpb: Int,
    dtype: DType,
    output_layout: Layout,
    x_layout: Layout,
    w_layout: Layout,
    b_layout: Layout
](
    output: LayoutTensor[dtype, output_layout, MutAnyOrigin],
    x: LayoutTensor[dtype, x_layout, ImmutAnyOrigin],
    w: LayoutTensor[dtype, w_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, ImmutAnyOrigin],
    batch_size: Int,
    output_dim: Int,
    input_dim: Int
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    
    blocks_per_batch = ((output_dim + tpb - 1) // tpb) * ((batch_size + tpb - 1) // tpb)
    block_id = block_idx.x + block_idx.y * grid_dim.x + block_idx.z * grid_dim.x * grid_dim.y
    
    blocks_x = (batch_size + tpb - 1) // tpb
    block_row = block_id // blocks_x
    block_col = block_id % blocks_x
    
    global_row = block_row * tpb + local_row
    global_col = block_col * tpb + local_col
    if global_row >= output_dim or global_col >= batch_size:
        return

    w_shared = LayoutTensor[
        dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    x_shared = LayoutTensor[
        dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    acc: output.element_type = 0
    for tile in range((input_dim + tpb - 1) // tpb):
        w_shared[local_row, local_col] = 0
        x_shared[local_row, local_col] = 0
        barrier()

        if global_row < output_dim and (tile * tpb + local_col) < input_dim:
            w_shared[local_row, local_col] = w[global_row, tile * tpb + local_col]
        if global_col < batch_size and (tile * tpb + local_row) < input_dim:
            x_shared[local_row, local_col] = x[global_col, tile * tpb + local_row]        
        barrier()

        @parameter
        for i in range(tpb):
            acc += w_shared[local_row, i] * x_shared[i, local_col]
        barrier()

    if global_row < output_dim and global_col < batch_size:
        output[global_col, global_row] = acc + rebind[Scalar[dtype]](b[global_row])


fn dense_backward[
    tpb: Int,
    dtype: DType,
    x_grad_layout: Layout,
    w_grad_layout: Layout,
    b_grad_layout: Layout,
    x_layout: Layout,
    w_layout: Layout,
    grad_output_layout: Layout
](
    x_gradient: LayoutTensor[dtype, x_grad_layout, MutAnyOrigin],
    w_gradient: LayoutTensor[dtype, w_grad_layout, MutAnyOrigin],
    b_gradient: LayoutTensor[dtype, b_grad_layout, MutAnyOrigin],
    x: LayoutTensor[dtype, x_layout, ImmutAnyOrigin],
    w: LayoutTensor[dtype, w_layout, ImmutAnyOrigin],
    grad_output: LayoutTensor[dtype, grad_output_layout, ImmutAnyOrigin],
    batch_size: Int,
    output_dim: Int,
    input_dim: Int
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    tid = local_row * tpb + local_col

    w_blocks = ((output_dim + tpb - 1) // tpb) * ((input_dim + tpb - 1) // tpb)
    b_blocks = (output_dim + tpb - 1) // tpb
    x_blocks_per_batch = ((batch_size + tpb - 1) // tpb) * ((input_dim + tpb - 1) // tpb)
    total_x_blocks = x_blocks_per_batch * 1
    
    block_id = block_idx.x + block_idx.y * grid_dim.x + block_idx.z * grid_dim.x * grid_dim.y
    
    # Weight gradient
    if block_id < w_blocks:
        w_blocks_x = (input_dim + tpb - 1) // tpb
        w_block_row = block_id // w_blocks_x
        w_block_col = block_id % w_blocks_x
        global_row = w_block_row * tpb + local_row
        global_col = w_block_col * tpb + local_col
        if global_row >= output_dim or global_col >= input_dim:
            return
        
        grad_output_shared = LayoutTensor[
            dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
            address_space = AddressSpace.SHARED
        ].stack_allocation()
        x_shared = LayoutTensor[
            dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
            address_space = AddressSpace.SHARED
        ].stack_allocation()
        
        acc: w_gradient.element_type = 0
        for batch in range(batch_size):
            grad_output_shared[local_row, local_col] = 0
            x_shared[local_row, local_col] = 0
            barrier()
            
            if global_row < output_dim and local_col == 0:
                grad_output_shared[local_row, 0] = grad_output[batch, global_row]
            if global_col < input_dim and local_row == 0:
                x_shared[0, local_col] = x[batch, global_col]
            barrier()
            
            acc += grad_output_shared[local_row, 0] * x_shared[0, local_col]
            barrier()
        
        w_gradient[global_row, global_col] = acc
        return
    
    block_id -= w_blocks
    
    # Bias gradient
    if block_id < b_blocks:
        feature_idx = block_id * tpb + tid
        if feature_idx >= output_dim:
            return
        
        acc: b_gradient.element_type = 0
        for batch in range(batch_size):
            acc += grad_output[batch, feature_idx]
        b_gradient[feature_idx] = acc
        return
    
    block_id -= b_blocks
    
    # Input gradient
    batch_blocks = (batch_size + tpb - 1) // tpb
    input_blocks = (input_dim + tpb - 1) // tpb
    
    x_block_row = block_id // input_blocks
    x_block_col = block_id % input_blocks
    
    global_row = x_block_row * tpb + local_row
    global_col = x_block_col * tpb + local_col
    if global_row >= batch_size or global_col >= input_dim:
        return
    
    grad_output_shared = LayoutTensor[
        dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()
    w_shared = LayoutTensor[
        dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()
    
    acc: x_gradient.element_type = 0
    for tile in range((output_dim + tpb - 1) // tpb):
        grad_output_shared[local_row, local_col] = 0
        w_shared[local_row, local_col] = 0
        barrier()

        if global_row < batch_size and (tile * tpb + local_col) < output_dim:
            grad_output_shared[local_row, local_col] = grad_output[global_row, tile * tpb + local_col]
        if global_col < input_dim and (tile * tpb + local_row) < output_dim:
            w_shared[local_row, local_col] = w[tile * tpb + local_row, global_col]
        barrier()
        
        @parameter
        for i in range(tpb):
            acc += grad_output_shared[local_row, i] * w_shared[i, local_col]
        barrier()
    
    x_gradient[global_row, global_col] = acc
