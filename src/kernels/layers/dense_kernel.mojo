from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
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
    size1: Int,
    size2: Int,
    size3: Int
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    
    blocks_per_batch = ((size3 + tpb - 1) // tpb) * ((size1 + tpb - 1) // tpb)
    block_id = block_idx.x + block_idx.y * grid_dim.x + block_idx.z * grid_dim.x * grid_dim.y
    batch_idx = block_id // blocks_per_batch
    batch_block_id = block_id % blocks_per_batch
    if batch_idx >= batch_size:
        return
    
    blocks_x = (size3 + tpb - 1) // tpb
    block_row = batch_block_id // blocks_x
    block_col = batch_block_id % blocks_x
    global_row = block_row * tpb + local_row
    global_col = block_col * tpb + local_col

    w_shared = LayoutTensor[
        dtype,
        Layout.row_major(tpb, tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    x_shared = LayoutTensor[
        dtype,
        Layout.row_major(tpb, tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    acc: output.element_type = 0

    for tile in range((size2 + tpb - 1) // tpb):
        w_shared[local_row, local_col] = 0
        x_shared[local_row, local_col] = 0
        barrier()

        if global_row < size1 and (tile * tpb + local_col) < size2:
            w_shared[local_row, local_col] = w[global_row, tile * tpb + local_col]
        if (tile * tpb + local_row) < size2 and global_col < size3:
            x_shared[local_row, local_col] = x[batch_idx, tile * tpb + local_row, global_col]        
        barrier()

        if global_row < size1 and global_col < size3:
            @parameter
            for i in range(tpb):
                acc += w_shared[local_row, i] * x_shared[i, local_col]
        barrier()

    if global_row < size1 and global_col < size3:
        output[batch_idx, global_row, global_col] = acc + rebind[Scalar[dtype]](b[global_row])

fn dense_backward[
    tpb: Int,
    dtype: DType,
    x_grad_layout: Layout,
    w_grad_layout: Layout,
    b_grad_layout: Layout,
    x_layout: Layout,
    w_layout: Layout,
    local_gradient_layout: Layout
](
    x_gradient: LayoutTensor[dtype, x_grad_layout, MutAnyOrigin],
    w_gradient: LayoutTensor[dtype, w_grad_layout, MutAnyOrigin],
    b_gradient: LayoutTensor[dtype, b_grad_layout, MutAnyOrigin],
    x: LayoutTensor[dtype, x_layout, ImmutAnyOrigin],
    w: LayoutTensor[dtype, w_layout, ImmutAnyOrigin],
    local_gradient: LayoutTensor[dtype, local_gradient_layout, ImmutAnyOrigin],
    batch_size: Int,
    size1: Int,
    size2: Int,
    size3: Int,
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    tid = local_row * tpb + local_col
    
    w_blocks = ((size1 + tpb - 1) // tpb) * ((size2 + tpb - 1) // tpb)
    b_blocks = (size1 + tpb - 1) // tpb
    x_blocks_per_batch = ((size2 + tpb - 1) // tpb) * ((size3 + tpb - 1) // tpb)
    
    block_id = block_idx.x + block_idx.y * grid_dim.x + block_idx.z * grid_dim.x * grid_dim.y
    
    # Weight gradient
    if block_id < w_blocks:
        w_blocks_x = (size2 + tpb - 1) // tpb
        w_block_row = block_id // w_blocks_x
        w_block_col = block_id % w_blocks_x
        global_row = w_block_row * tpb + local_row
        global_col = w_block_col * tpb + local_col
        if global_row >= size1 or global_col >= size2:
            return
        
        local_gradient_shared = LayoutTensor[
            dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
            address_space = AddressSpace.SHARED
        ].stack_allocation()
        x_shared = LayoutTensor[
            dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
            address_space = AddressSpace.SHARED
        ].stack_allocation()
        
        acc: w_gradient.element_type = 0
        for batch in range(batch_size):
            for tile in range((size3 + tpb - 1) // tpb):
                local_gradient_shared[local_row, local_col] = 0
                x_shared[local_row, local_col] = 0
                barrier()
                
                if global_row < size1 and (tile * tpb + local_col) < size3:
                    local_gradient_shared[local_row, local_col] = local_gradient[
                        batch, global_row, tile * tpb + local_col
                    ]
                if global_col < size2 and (tile * tpb + local_row) < size3:
                    x_shared[local_row, local_col] = x[
                        batch, global_col, tile * tpb + local_row
                    ]
                barrier()

                @parameter
                for j in range(tpb):
                    acc += local_gradient_shared[local_row, j] * x_shared[j, local_col]
                barrier()
        w_gradient[global_row, global_col] = acc
        return
    block_id -= w_blocks
    
    # Bias gradient
    if block_id < b_blocks:
        feature_idx = block_id * tpb + tid
        if feature_idx >= size1:
            return
        
        acc: b_gradient.element_type = 0
        for batch in range(batch_size):
            for k in range(size3):
                acc += local_gradient[batch, feature_idx, k]
        b_gradient[feature_idx] = acc
        return
    block_id -= b_blocks
    
    # Input gradient
    batch = block_id // x_blocks_per_batch
    batch_block_id = block_id % x_blocks_per_batch
    if batch >= batch_size:
        return
    
    x_blocks_x = (size3 + tpb - 1) // tpb
    x_block_row = batch_block_id // x_blocks_x
    x_block_col = batch_block_id % x_blocks_x
    global_row = x_block_row * tpb + local_row
    global_col = x_block_col * tpb + local_col
    if global_row >= size2 or global_col >= size3:
        return
    
    w_shared = LayoutTensor[
        dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()
    local_gradient_shared = LayoutTensor[
        dtype, Layout.row_major(tpb, tpb), MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()
    
    acc: x_gradient.element_type = 0
    for tile in range((size1 + tpb - 1) // tpb):
        w_shared[local_row, local_col] = 0
        local_gradient_shared[local_row, local_col] = 0
        barrier()
        
        if (tile * tpb + local_col) < size1 and global_row < size2:
            w_shared[local_row, local_col] = w[tile * tpb + local_col, global_row]
        if (tile * tpb + local_row) < size1 and global_col < size3:
            local_gradient_shared[local_row, local_col] = local_gradient[
                batch, tile * tpb + local_row, global_col
            ]
        barrier()
        
        @parameter
        for j in range(tpb):
            acc += w_shared[local_row, j] * local_gradient_shared[j, local_col]    
        barrier()
    x_gradient[batch, global_row, global_col] = acc
