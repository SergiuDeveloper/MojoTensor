from random import random_float64, seed
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from testing import TestSuite, assert_almost_equal

from src.kernels.losses import mse_forward, mse_backward
from src.kernels.constants import MAX_GRID_SIZE

fn test_mse_forward() raises:
    comptime TPB = 16
    comptime DTYPE = DType.float64
    comptime OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime PREDICTIONS_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime TARGETS_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)

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
        predictions = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        predictions.enqueue_fill(0)
        targets = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        targets.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        expected.enqueue_fill(0)

        with predictions.map_to_host() as predictions_host, targets.map_to_host() as targets_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        predictions_host[batch * SIZE1 * SIZE2 + row * SIZE2 + col] = random_float64()
                        targets_host[batch * SIZE1 * SIZE2 + row * SIZE2 + col] = random_float64()
            
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        idx = batch * SIZE1 * SIZE2 + row * SIZE2 + col
                        diff = predictions_host[idx] - targets_host[idx]
                        expected[idx] = diff * diff

        output_tensor = LayoutTensor[DTYPE, OUTPUT_LAYOUT, MutAnyOrigin](
            output,
            RuntimeLayout[OUTPUT_LAYOUT](
                RuntimeTuple[OUTPUT_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[OUTPUT_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )
        predictions_tensor = LayoutTensor[DTYPE, PREDICTIONS_LAYOUT, ImmutAnyOrigin](
            predictions,
            RuntimeLayout[PREDICTIONS_LAYOUT](
                RuntimeTuple[PREDICTIONS_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[PREDICTIONS_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )
        targets_tensor = LayoutTensor[DTYPE, TARGETS_LAYOUT, ImmutAnyOrigin](
            targets,
            RuntimeLayout[TARGETS_LAYOUT](
                RuntimeTuple[TARGETS_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[TARGETS_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )

        comptime kernel = mse_forward[TPB, DTYPE, OUTPUT_LAYOUT, PREDICTIONS_LAYOUT, TARGETS_LAYOUT]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            predictions_tensor,
            targets_tensor,
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


fn test_mse_backward() raises:
    comptime TPB = 16
    comptime DTYPE = DType.float64
    comptime OUTPUT_INPUT_GRADIENT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime PREDICTIONS_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime TARGETS_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
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
        output_input_gradient = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        output_input_gradient.enqueue_fill(0)
        predictions = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        predictions.enqueue_fill(0)
        targets = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        targets.enqueue_fill(0)
        local_gradient = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        local_gradient.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * SIZE1 * SIZE2)
        expected.enqueue_fill(0)

        with predictions.map_to_host() as predictions_host, targets.map_to_host() as targets_host, local_gradient.map_to_host() as local_gradient_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        predictions_host[batch * SIZE1 * SIZE2 + row * SIZE2 + col] = random_float64()
                        targets_host[batch * SIZE1 * SIZE2 + row * SIZE2 + col] = random_float64()
                        local_gradient_host[batch * SIZE1 * SIZE2 + row * SIZE2 + col] = random_float64()
            
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        idx = batch * SIZE1 * SIZE2 + row * SIZE2 + col
                        expected[idx] = 2 * (predictions_host[idx] - targets_host[idx]) * local_gradient_host[idx]

        output_input_gradient_tensor = LayoutTensor[DTYPE, OUTPUT_INPUT_GRADIENT_LAYOUT, MutAnyOrigin](
            output_input_gradient,
            RuntimeLayout[OUTPUT_INPUT_GRADIENT_LAYOUT](
                RuntimeTuple[OUTPUT_INPUT_GRADIENT_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[OUTPUT_INPUT_GRADIENT_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )
        predictions_tensor = LayoutTensor[DTYPE, PREDICTIONS_LAYOUT, ImmutAnyOrigin](
            predictions,
            RuntimeLayout[PREDICTIONS_LAYOUT](
                RuntimeTuple[PREDICTIONS_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[PREDICTIONS_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )
        targets_tensor = LayoutTensor[DTYPE, TARGETS_LAYOUT, ImmutAnyOrigin](
            targets,
            RuntimeLayout[TARGETS_LAYOUT](
                RuntimeTuple[TARGETS_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[TARGETS_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )
        local_gradient_tensor = LayoutTensor[DTYPE, LOCAL_GRADIENT_LAYOUT, ImmutAnyOrigin](
            local_gradient,
            RuntimeLayout[LOCAL_GRADIENT_LAYOUT](
                RuntimeTuple[LOCAL_GRADIENT_LAYOUT.shape](BATCH_SIZE, SIZE1, SIZE2),
                RuntimeTuple[LOCAL_GRADIENT_LAYOUT.stride](SIZE1 * SIZE2, SIZE2, 1)
            )
        )

        comptime kernel = mse_backward[TPB, DTYPE, OUTPUT_INPUT_GRADIENT_LAYOUT, PREDICTIONS_LAYOUT, TARGETS_LAYOUT, LOCAL_GRADIENT_LAYOUT]
        ctx.enqueue_function[kernel, kernel](
            output_input_gradient_tensor,
            predictions_tensor,
            targets_tensor,
            local_gradient_tensor,
            BATCH_SIZE,
            SIZE1,
            SIZE2,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK
        )
        ctx.synchronize()

        with output_input_gradient.map_to_host() as output_input_gradient_host:
            for batch in range(BATCH_SIZE):
                for row in range(SIZE1):
                    for col in range(SIZE2):
                        idx = batch * SIZE1 * SIZE2 + row * SIZE2 + col
                        assert_almost_equal(output_input_gradient_host[idx], expected[idx], rtol=1e-10)


fn main() raises:
    seed(42)
    TestSuite.discover_tests[__functions_in_module()]().run()
