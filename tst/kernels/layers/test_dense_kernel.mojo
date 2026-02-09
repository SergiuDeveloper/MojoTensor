from random import random_float64, seed
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from testing import TestSuite, assert_almost_equal

from src.kernels.layers import dense_forward, dense_backward
from src.kernels.constants import MAX_GRID_SIZE

fn test_dense_forward() raises:
    comptime TPB = 16
    comptime DTYPE = DType.float64
    comptime OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime W_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime B_LAYOUT = Layout.row_major(UNKNOWN_VALUE)

    BATCH_SIZE = 32
    OUTPUT_DIM = 64
    INPUT_DIM = 128
    
    output_blocks = (OUTPUT_DIM + TPB - 1) // TPB
    batch_blocks = (BATCH_SIZE + TPB - 1) // TPB
    total_blocks = output_blocks * batch_blocks
    
    grid_x = min(total_blocks, MAX_GRID_SIZE)
    grid_y = (total_blocks + grid_x - 1) // grid_x
    grid_z = 1
    
    BLOCKS_PER_GRID = (grid_x, grid_y, grid_z)
    THREADS_PER_BLOCK = (TPB, TPB)

    with DeviceContext() as ctx:
        output = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_DIM)
        output.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * INPUT_DIM)
        x.enqueue_fill(0)
        w = ctx.enqueue_create_buffer[DTYPE](OUTPUT_DIM * INPUT_DIM)
        w.enqueue_fill(0)
        b = ctx.enqueue_create_buffer[DTYPE](OUTPUT_DIM)
        b.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * OUTPUT_DIM)
        expected.enqueue_fill(0)

        with x.map_to_host() as x_host, w.map_to_host() as w_host, b.map_to_host() as b_host:
            for batch in range(BATCH_SIZE):
                for col in range(INPUT_DIM):
                    x_host[batch * INPUT_DIM + col] = random_float64()
            for row in range(OUTPUT_DIM):
                for col in range(INPUT_DIM):
                    w_host[row * INPUT_DIM + col] = random_float64()
            for row in range(OUTPUT_DIM):
                b_host[row] = random_float64()
            
            for batch in range(BATCH_SIZE):
                for i in range(OUTPUT_DIM):
                    acc: Float64 = b_host[i]
                    for k in range(INPUT_DIM):
                        acc += w_host[i * INPUT_DIM + k] * x_host[batch * INPUT_DIM + k]
                    expected[batch * OUTPUT_DIM + i] = acc

        output_tensor = LayoutTensor[DTYPE, OUTPUT_LAYOUT, MutAnyOrigin](
            output,
            RuntimeLayout[OUTPUT_LAYOUT](
                RuntimeTuple[OUTPUT_LAYOUT.shape](BATCH_SIZE, OUTPUT_DIM),
                RuntimeTuple[OUTPUT_LAYOUT.stride](OUTPUT_DIM, 1)
            )
        )
        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, ImmutAnyOrigin](
            x,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, INPUT_DIM),
                RuntimeTuple[X_LAYOUT.stride](INPUT_DIM, 1)
            )
        )
        w_tensor = LayoutTensor[DTYPE, W_LAYOUT, ImmutAnyOrigin](
            w,
            RuntimeLayout[W_LAYOUT](
                RuntimeTuple[W_LAYOUT.shape](OUTPUT_DIM, INPUT_DIM),
                RuntimeTuple[W_LAYOUT.stride](INPUT_DIM, 1)
            )
        )
        b_tensor = LayoutTensor[DTYPE, B_LAYOUT, ImmutAnyOrigin](
            b,
            RuntimeLayout[B_LAYOUT](
                RuntimeTuple[B_LAYOUT.shape](OUTPUT_DIM),
                RuntimeTuple[B_LAYOUT.stride](1)
            )
        )

        comptime kernel = dense_forward[
            TPB,
            DTYPE,
            OUTPUT_LAYOUT,
            X_LAYOUT,
            W_LAYOUT,
            B_LAYOUT
        ]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            x_tensor,
            w_tensor,
            b_tensor,
            BATCH_SIZE,
            OUTPUT_DIM,
            INPUT_DIM,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()

        with output.map_to_host() as output_host:
            for batch in range(BATCH_SIZE):
                for row in range(OUTPUT_DIM):
                    idx = batch * OUTPUT_DIM + row
                    assert_almost_equal(output_host[idx], expected[idx], rtol=1e-10)

fn test_dense_backward() raises:
    comptime TPB = 16
    comptime DTYPE = DType.float64
    comptime X_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime W_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime B_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE)
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime W_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime GRAD_OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    BATCH_SIZE = 32
    OUTPUT_DIM = 64
    INPUT_DIM = 128

    w_blocks = ((OUTPUT_DIM + TPB - 1) // TPB) * ((INPUT_DIM + TPB - 1) // TPB)
    b_blocks = (OUTPUT_DIM + TPB - 1) // TPB
    x_blocks = ((BATCH_SIZE + TPB - 1) // TPB) * ((INPUT_DIM + TPB - 1) // TPB)
    total_blocks = w_blocks + b_blocks + x_blocks

    grid_x = min(total_blocks, MAX_GRID_SIZE)
    grid_y = (total_blocks + grid_x - 1) // grid_x
    grid_z = 1

    BLOCKS_PER_GRID = (grid_x, grid_y, grid_z)
    THREADS_PER_BLOCK = (TPB, TPB)

    with DeviceContext() as ctx:
        w_gradient = ctx.enqueue_create_buffer[DTYPE](OUTPUT_DIM * INPUT_DIM)
        w_gradient.enqueue_fill(0)
        b_gradient = ctx.enqueue_create_buffer[DTYPE](OUTPUT_DIM)
        b_gradient.enqueue_fill(0)
        x_gradient = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * INPUT_DIM)
        x_gradient.enqueue_fill(0)
        
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * INPUT_DIM)
        x.enqueue_fill(0)
        w = ctx.enqueue_create_buffer[DTYPE](OUTPUT_DIM * INPUT_DIM)
        w.enqueue_fill(0)
        grad_output = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_DIM)
        grad_output.enqueue_fill(0)
        
        expected_w_gradient = ctx.enqueue_create_host_buffer[DTYPE](OUTPUT_DIM * INPUT_DIM)
        expected_w_gradient.enqueue_fill(0)
        expected_b_gradient = ctx.enqueue_create_host_buffer[DTYPE](OUTPUT_DIM)
        expected_b_gradient.enqueue_fill(0)
        expected_x_gradient = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * INPUT_DIM)
        expected_x_gradient.enqueue_fill(0)
        
        with x.map_to_host() as x_host, w.map_to_host() as w_host, grad_output.map_to_host() as grad_output_host:
            for batch in range(BATCH_SIZE):
                for col in range(INPUT_DIM):
                    x_host[batch * INPUT_DIM + col] = random_float64()
            
            for row in range(OUTPUT_DIM):
                for col in range(INPUT_DIM):
                    w_host[row * INPUT_DIM + col] = random_float64()
            
            for batch in range(BATCH_SIZE):
                for row in range(OUTPUT_DIM):
                    grad_output_host[batch * OUTPUT_DIM + row] = random_float64()
            
            for i in range(OUTPUT_DIM):
                for k in range(INPUT_DIM):
                    acc: Float64 = 0.0
                    for batch in range(BATCH_SIZE):
                        acc += grad_output_host[batch * OUTPUT_DIM + i] * x_host[batch * INPUT_DIM + k]
                    expected_w_gradient[i * INPUT_DIM + k] = acc
            
            for i in range(OUTPUT_DIM):
                acc: Float64 = 0.0
                for batch in range(BATCH_SIZE):
                    acc += grad_output_host[batch * OUTPUT_DIM + i]
                expected_b_gradient[i] = acc
            
            for batch in range(BATCH_SIZE):
                for k in range(INPUT_DIM):
                    acc: Float64 = 0.0
                    for i in range(OUTPUT_DIM):
                        acc += w_host[i * INPUT_DIM + k] * grad_output_host[batch * OUTPUT_DIM + i]
                    expected_x_gradient[batch * INPUT_DIM + k] = acc

        w_gradient_tensor = LayoutTensor[DTYPE, W_GRAD_LAYOUT, MutAnyOrigin](
            w_gradient,
            RuntimeLayout[W_GRAD_LAYOUT](
                RuntimeTuple[W_GRAD_LAYOUT.shape](OUTPUT_DIM, INPUT_DIM),
                RuntimeTuple[W_GRAD_LAYOUT.stride](INPUT_DIM, 1)
            )
        )
        b_gradient_tensor = LayoutTensor[DTYPE, B_GRAD_LAYOUT, MutAnyOrigin](
            b_gradient,
            RuntimeLayout[B_GRAD_LAYOUT](
                RuntimeTuple[B_GRAD_LAYOUT.shape](OUTPUT_DIM),
                RuntimeTuple[B_GRAD_LAYOUT.stride](1)
            )
        )
        x_gradient_tensor = LayoutTensor[DTYPE, X_GRAD_LAYOUT, MutAnyOrigin](
            x_gradient,
            RuntimeLayout[X_GRAD_LAYOUT](
                RuntimeTuple[X_GRAD_LAYOUT.shape](BATCH_SIZE, INPUT_DIM),
                RuntimeTuple[X_GRAD_LAYOUT.stride](INPUT_DIM, 1)
            )
        )
        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, ImmutAnyOrigin](
            x,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, INPUT_DIM),
                RuntimeTuple[X_LAYOUT.stride](INPUT_DIM, 1)
            )
        )
        w_tensor = LayoutTensor[DTYPE, W_LAYOUT, ImmutAnyOrigin](
            w,
            RuntimeLayout[W_LAYOUT](
                RuntimeTuple[W_LAYOUT.shape](OUTPUT_DIM, INPUT_DIM),
                RuntimeTuple[W_LAYOUT.stride](INPUT_DIM, 1)
            )
        )
        grad_output_tensor = LayoutTensor[DTYPE, GRAD_OUTPUT_LAYOUT, ImmutAnyOrigin](
            grad_output,
            RuntimeLayout[GRAD_OUTPUT_LAYOUT](
                RuntimeTuple[GRAD_OUTPUT_LAYOUT.shape](BATCH_SIZE, OUTPUT_DIM),
                RuntimeTuple[GRAD_OUTPUT_LAYOUT.stride](OUTPUT_DIM, 1)
            )
        )

        comptime kernel = dense_backward[
            TPB,
            DTYPE,
            X_GRAD_LAYOUT,
            W_GRAD_LAYOUT,
            B_GRAD_LAYOUT,
            X_LAYOUT,
            W_LAYOUT,
            GRAD_OUTPUT_LAYOUT
        ]
        ctx.enqueue_function[kernel, kernel](
            x_gradient_tensor,
            w_gradient_tensor,
            b_gradient_tensor,
            x_tensor,
            w_tensor,
            grad_output_tensor,
            BATCH_SIZE,
            OUTPUT_DIM,
            INPUT_DIM,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()

        with w_gradient.map_to_host() as w_gradient_host:
            for row in range(OUTPUT_DIM):
                for col in range(INPUT_DIM):
                    idx = row * INPUT_DIM + col
                    assert_almost_equal(w_gradient_host[idx], expected_w_gradient[idx], rtol=1e-10)
        
        with b_gradient.map_to_host() as b_gradient_host:
            for i in range(OUTPUT_DIM):
                assert_almost_equal(b_gradient_host[i], expected_b_gradient[i], rtol=1e-10)
        
        with x_gradient.map_to_host() as x_gradient_host:
            for batch in range(BATCH_SIZE):
                for col in range(INPUT_DIM):
                    idx = batch * INPUT_DIM + col
                    assert_almost_equal(x_gradient_host[idx], expected_x_gradient[idx], rtol=1e-10)

fn main() raises:
    seed(42)
    TestSuite.discover_tests[__functions_in_module()]().run()
