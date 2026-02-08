from random import random_float64, seed
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from testing import TestSuite, assert_almost_equal

from src.kernels.layers import dense_forward, dense_backward
from src.kernels.constants import MAX_GRID_SIZE

def test_dense_forward():
    comptime TPB = 16
    comptime DTYPE = DType.float64
    comptime OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime W_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime B_LAYOUT = Layout.row_major(UNKNOWN_VALUE)

    BATCH_SIZE = 32
    SIZE1 = 64
    SIZE2 = 128
    SIZE3 = 256
    
    blocks_per_batch = ((SIZE3 + TPB - 1) // TPB) * ((SIZE1 + TPB - 1) // TPB)
    total_blocks = BATCH_SIZE * blocks_per_batch
    
    grid_x = min(total_blocks, MAX_GRID_SIZE)
    grid_y = (total_blocks + grid_x - 1) // grid_x
    grid_z = 1
    
    BLOCKS_PER_GRID = (grid_x, grid_y, grid_z)
    THREADS_PER_BLOCK = (TPB, TPB)

    with DeviceContext() as ctx:
        output = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE3)
        output.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE2 * SIZE3)
        x.enqueue_fill(0)
        w = ctx.enqueue_create_buffer[DTYPE](SIZE1 * SIZE2)
        w.enqueue_fill(0)
        b = ctx.enqueue_create_buffer[DTYPE](SIZE1)
        b.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE3)
        expected.enqueue_fill(0)

        with x.map_to_host() as x_host, w.map_to_host() as w_host, b.map_to_host() as b_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE2):
                    for col in range(SIZE3):
                        x_host[batch * SIZE2 * SIZE3 + row * SIZE3 + col] = random_float64()
            for row in range(SIZE1):
                for col in range(SIZE2):
                    w_host[row * SIZE2 + col] = random_float64()
            for row in range(SIZE1):
                b_host[row] = random_float64()
            
            for batch in range(BATCH_SIZE):
                for i in range(SIZE1):
                    for j in range(SIZE3):
                        acc: Float64 = b_host[i]
                        for k in range(SIZE2):
                            acc += w_host[i * SIZE2 + k] * x_host[batch * SIZE2 * SIZE3 + k * SIZE3 + j]
                        expected[batch * SIZE1 * SIZE3 + i * SIZE3 + j] = acc

        output_tensor = LayoutTensor[DTYPE, OUTPUT_LAYOUT, MutAnyOrigin](
            output,
            RuntimeLayout[OUTPUT_LAYOUT](
                RuntimeTuple[OUTPUT_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE3),
                RuntimeTuple[OUTPUT_LAYOUT.stride](SIZE1 * SIZE3, SIZE3, 1)
            )
        )
        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, ImmutAnyOrigin](
            x,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, SIZE2, SIZE3),
                RuntimeTuple[X_LAYOUT.stride](SIZE2 * SIZE3, SIZE3, 1)
            )
        )
        w_tensor = LayoutTensor[DTYPE, W_LAYOUT, ImmutAnyOrigin](
            w,
            RuntimeLayout[W_LAYOUT](
                RuntimeTuple[W_LAYOUT.shape](SIZE1, SIZE2),
                RuntimeTuple[W_LAYOUT.stride](SIZE2, 1)
            )
        )
        b_tensor = LayoutTensor[DTYPE, B_LAYOUT, ImmutAnyOrigin](
            b,
            RuntimeLayout[B_LAYOUT](
                RuntimeTuple[B_LAYOUT.shape](SIZE1),
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
            SIZE1,
            SIZE2,
            SIZE3,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()

        with output.map_to_host() as output_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE3):
                        idx = batch * SIZE1 * SIZE3 + row * SIZE3 + col
                        assert_almost_equal(output_host[idx], expected[idx], rtol=1e-10)

def test_dense_backward():
    comptime TPB = 16
    comptime DTYPE = DType.float64
    comptime X_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime W_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime B_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE)
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime W_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime LOCAL_GRADIENT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)

    BATCH_SIZE = 32
    SIZE1 = 64
    SIZE2 = 128
    SIZE3 = 256

    w_blocks = ((SIZE1 + TPB - 1) // TPB) * ((SIZE2 + TPB - 1) // TPB)
    b_blocks = (SIZE1 + TPB - 1) // TPB
    x_blocks_per_batch = ((SIZE2 + TPB - 1) // TPB) * ((SIZE3 + TPB - 1) // TPB)
    x_blocks = BATCH_SIZE * x_blocks_per_batch
    total_blocks = w_blocks + b_blocks + x_blocks

    grid_x = min(total_blocks, MAX_GRID_SIZE)
    grid_y = (total_blocks + grid_x - 1) // grid_x
    grid_z = 1

    BLOCKS_PER_GRID = (grid_x, grid_y, grid_z)
    THREADS_PER_BLOCK = (TPB, TPB)

    with DeviceContext() as ctx:
        w_gradient = ctx.enqueue_create_buffer[DTYPE](SIZE1 * SIZE2)
        w_gradient.enqueue_fill(0)
        b_gradient = ctx.enqueue_create_buffer[DTYPE](SIZE1)
        b_gradient.enqueue_fill(0)
        x_gradient = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE2 * SIZE3)
        x_gradient.enqueue_fill(0)
        
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE2 * SIZE3)
        x.enqueue_fill(0)
        w = ctx.enqueue_create_buffer[DTYPE](SIZE1 * SIZE2)
        w.enqueue_fill(0)
        local_gradient = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE3)
        local_gradient.enqueue_fill(0)
        
        expected_w = ctx.enqueue_create_host_buffer[DTYPE](SIZE1 * SIZE2)
        expected_w.enqueue_fill(0)
        expected_b = ctx.enqueue_create_host_buffer[DTYPE](SIZE1)
        expected_b.enqueue_fill(0)
        expected_x = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * SIZE2 * SIZE3)
        expected_x.enqueue_fill(0)
        
        with x.map_to_host() as x_host, w.map_to_host() as w_host, local_gradient.map_to_host() as local_gradient_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE2):
                    for col in range(SIZE3):
                        x_host[batch * SIZE2 * SIZE3 + row * SIZE3 + col] = random_float64()
            
            for row in range(SIZE1):
                for col in range(SIZE2):
                    w_host[row * SIZE2 + col] = random_float64()
            
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE3):
                        local_gradient_host[batch * SIZE1 * SIZE3 + row * SIZE3 + col] = random_float64()
            
            for i in range(SIZE1):
                for k in range(SIZE2):
                    acc: Float64 = 0.0
                    for batch in range(BATCH_SIZE):
                        for j in range(SIZE3):
                            acc += local_gradient_host[batch * SIZE1 * SIZE3 + i * SIZE3 + j] * x_host[batch * SIZE2 * SIZE3 + k * SIZE3 + j]
                    expected_w[i * SIZE2 + k] = acc
            
            for i in range(SIZE1):
                acc: Float64 = 0.0
                for batch in range(BATCH_SIZE):
                    for j in range(SIZE3):
                        acc += local_gradient_host[batch * SIZE1 * SIZE3 + i * SIZE3 + j]
                expected_b[i] = acc
            
            for batch in range(BATCH_SIZE):
                for k in range(SIZE2):
                    for j in range(SIZE3):
                        acc: Float64 = 0.0
                        for i in range(SIZE1):
                            acc += w_host[i * SIZE2 + k] * local_gradient_host[batch * SIZE1 * SIZE3 + i * SIZE3 + j]
                        expected_x[batch * SIZE2 * SIZE3 + k * SIZE3 + j] = acc

        w_gradient_tensor = LayoutTensor[DTYPE, W_GRAD_LAYOUT, MutAnyOrigin](
            w_gradient,
            RuntimeLayout[W_GRAD_LAYOUT](
                RuntimeTuple[W_GRAD_LAYOUT.shape](SIZE1, SIZE2),
                RuntimeTuple[W_GRAD_LAYOUT.stride](SIZE2, 1)
            )
        )
        b_gradient_tensor = LayoutTensor[DTYPE, B_GRAD_LAYOUT, MutAnyOrigin](
            b_gradient,
            RuntimeLayout[B_GRAD_LAYOUT](
                RuntimeTuple[B_GRAD_LAYOUT.shape](SIZE1),
                RuntimeTuple[B_GRAD_LAYOUT.stride](1)
            )
        )
        x_gradient_tensor = LayoutTensor[DTYPE, X_GRAD_LAYOUT, MutAnyOrigin](
            x_gradient,
            RuntimeLayout[X_GRAD_LAYOUT](
                RuntimeTuple[X_GRAD_LAYOUT.shape](BATCH_SIZE, SIZE2, SIZE3),
                RuntimeTuple[X_GRAD_LAYOUT.stride](SIZE2 * SIZE3, SIZE3, 1)
            )
        )
        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, ImmutAnyOrigin](
            x,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, SIZE2, SIZE3),
                RuntimeTuple[X_LAYOUT.stride](SIZE2 * SIZE3, SIZE3, 1)
            )
        )
        w_tensor = LayoutTensor[DTYPE, W_LAYOUT, ImmutAnyOrigin](
            w,
            RuntimeLayout[W_LAYOUT](
                RuntimeTuple[W_LAYOUT.shape](SIZE1, SIZE2),
                RuntimeTuple[W_LAYOUT.stride](SIZE2, 1)
            )
        )
        local_gradient_tensor = LayoutTensor[DTYPE, LOCAL_GRADIENT_LAYOUT, ImmutAnyOrigin](
            local_gradient,
            RuntimeLayout[LOCAL_GRADIENT_LAYOUT](
                RuntimeTuple[LOCAL_GRADIENT_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE3),
                RuntimeTuple[LOCAL_GRADIENT_LAYOUT.stride](SIZE1 * SIZE3, SIZE3, 1)
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
            LOCAL_GRADIENT_LAYOUT
        ]
        ctx.enqueue_function[kernel, kernel](
            x_gradient_tensor,
            w_gradient_tensor,
            b_gradient_tensor,
            x_tensor,
            w_tensor,
            local_gradient_tensor,
            BATCH_SIZE,
            SIZE1,
            SIZE2,
            SIZE3,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()

        # Verify weight gradient
        with w_gradient.map_to_host() as w_gradient_host:
            for row in range(SIZE1):
                for col in range(SIZE2):
                    idx = row * SIZE2 + col
                    assert_almost_equal(w_gradient_host[idx], expected_w[idx], rtol=1e-10)
        
        # Verify bias gradient
        with b_gradient.map_to_host() as b_gradient_host:
            for i in range(SIZE1):
                assert_almost_equal(b_gradient_host[i], expected_b[i], rtol=1e-10)
        
        # Verify input gradient
        with x_gradient.map_to_host() as x_gradient_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE2):
                    for col in range(SIZE3):
                        idx = batch * SIZE2 * SIZE3 + row * SIZE3 + col
                        assert_almost_equal(x_gradient_host[idx], expected_x[idx], rtol=1e-10)

def main():
    seed(42)
    TestSuite.discover_tests[__functions_in_module()]().run()
