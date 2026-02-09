from random import random_float64, seed
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from testing import TestSuite, assert_almost_equal

from src.kernels.activations import relu_forward, relu_backward
from src.kernels.constants import MAX_GRID_SIZE

def test_relu_forward():
    comptime TPB = 16
    comptime DTYPE = DType.float64
    comptime OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)

    BATCH_SIZE = 32
    SIZE1 = 128
    SIZE2 = 256
    
    blocks_per_batch = ((SIZE2 + TPB - 1) // TPB) * ((SIZE1 + TPB - 1) // TPB)
    total_blocks = BATCH_SIZE * blocks_per_batch
    
    grid_x = min(total_blocks, MAX_GRID_SIZE)
    grid_y = (total_blocks + grid_x - 1) // grid_x
    grid_z = 1
    
    BLOCKS_PER_GRID = (grid_x, grid_y, grid_z)
    THREADS_PER_BLOCK = (TPB, TPB)

    with DeviceContext() as ctx:
        output = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        output.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        x.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        expected.enqueue_fill(0)

        with x.map_to_host() as x_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        x_host[batch * SIZE1 * SIZE2 + row * SIZE2 + col] = random_float64() - 0.5
            
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        idx = batch * SIZE1 * SIZE2 + row * SIZE2 + col
                        expected[idx] = max(x_host[idx], 0.0)

        output_tensor = LayoutTensor[DTYPE, OUTPUT_LAYOUT, MutAnyOrigin](
            output,
            RuntimeLayout[OUTPUT_LAYOUT](
                RuntimeTuple[OUTPUT_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[OUTPUT_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )
        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, ImmutAnyOrigin](
            x,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[X_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )

        comptime kernel = relu_forward[TPB, DTYPE, OUTPUT_LAYOUT, X_LAYOUT]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            x_tensor,
            BATCH_SIZE,
            SIZE1,
            SIZE2,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()

        with output.map_to_host() as output_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        idx = batch * SIZE1 * SIZE2 + row * SIZE2 + col
                        assert_almost_equal(output_host[idx], expected[idx], rtol=1e-10)


def test_relu_backward():
    comptime TPB = 16
    comptime DTYPE = DType.float64
    comptime X_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime LOCAL_GRADIENT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)

    BATCH_SIZE = 32
    SIZE1 = 128
    SIZE2 = 256
    
    blocks_per_batch = ((SIZE2 + TPB - 1) // TPB) * ((SIZE1 + TPB - 1) // TPB)
    total_blocks = BATCH_SIZE * blocks_per_batch
    
    grid_x = min(total_blocks, MAX_GRID_SIZE)
    grid_y = (total_blocks + grid_x - 1) // grid_x
    grid_z = 1
    
    BLOCKS_PER_GRID = (grid_x, grid_y, grid_z)
    THREADS_PER_BLOCK = (TPB, TPB)

    with DeviceContext() as ctx:
        x_gradient = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        x_gradient.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        x.enqueue_fill(0)
        local_gradient = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        local_gradient.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        expected.enqueue_fill(0)

        with x.map_to_host() as x_host, local_gradient.map_to_host() as local_gradient_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        x_host[batch * SIZE1 * SIZE2 + row * SIZE2 + col] = random_float64() - 0.5
                        local_gradient_host[batch * SIZE1 * SIZE2 + row * SIZE2 + col] = random_float64()
            
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        idx = batch * SIZE1 * SIZE2 + row * SIZE2 + col
                        expected[idx] = local_gradient_host[idx] if x_host[idx] > 0 else 0.0

        x_gradient_tensor = LayoutTensor[DTYPE, X_GRAD_LAYOUT, MutAnyOrigin](
            x_gradient,
            RuntimeLayout[X_GRAD_LAYOUT](
                RuntimeTuple[X_GRAD_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[X_GRAD_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )
        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, ImmutAnyOrigin](
            x,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[X_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )
        local_gradient_tensor = LayoutTensor[DTYPE, LOCAL_GRADIENT_LAYOUT, ImmutAnyOrigin](
            local_gradient,
            RuntimeLayout[LOCAL_GRADIENT_LAYOUT](
                RuntimeTuple[LOCAL_GRADIENT_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[LOCAL_GRADIENT_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )

        comptime kernel = relu_backward[TPB, DTYPE, X_GRAD_LAYOUT, X_LAYOUT, LOCAL_GRADIENT_LAYOUT]
        ctx.enqueue_function[kernel, kernel](
            x_gradient_tensor,
            x_tensor,
            local_gradient_tensor,
            BATCH_SIZE,
            SIZE1,
            SIZE2,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()

        with x_gradient.map_to_host() as x_gradient_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        idx = batch * SIZE1 * SIZE2 + row * SIZE2 + col
                        assert_almost_equal(x_gradient_host[idx], expected[idx], rtol=1e-10)

def main():
    seed(42)
    TestSuite.discover_tests[__functions_in_module()]().run()
