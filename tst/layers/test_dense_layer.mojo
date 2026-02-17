from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from random import random_float64
from testing import TestSuite, assert_almost_equal

from src.layers import Dense
from src.computational_graph import ComputationalGraph

fn test_dense_forward() raises:
    comptime BATCH_SIZE = 32
    comptime INPUT_NEURONS = 128
    comptime OUTPUT_NEURONS = 64
    comptime DTYPE = DType.float64
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    computational_graph = ComputationalGraph[DTYPE]()
    computational_graph_ptr = UnsafePointer(to=computational_graph)
    dense_layer = Dense[DTYPE](computational_graph_ptr, INPUT_NEURONS, OUTPUT_NEURONS)

    with DeviceContext() as ctx:
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * INPUT_NEURONS)
        x.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * OUTPUT_NEURONS)
        expected.enqueue_fill(0)

        with x.map_to_host() as x_host:
            for batch in range(BATCH_SIZE):
                for col in range(INPUT_NEURONS):
                    x_host[batch * INPUT_NEURONS + col] = random_float64()

            w_ptr = dense_layer.w_cpu.unsafe_ptr()
            b_ptr = dense_layer.b_cpu.unsafe_ptr()
            for batch in range(BATCH_SIZE):
                for i in range(OUTPUT_NEURONS):
                    acc: Float64 = b_ptr[i].cast[DType.float64]()
                    for k in range(INPUT_NEURONS):
                        w_val = w_ptr[i * INPUT_NEURONS + k].cast[DType.float64]()
                        x_val = x_host[batch * INPUT_NEURONS + k]
                        acc += w_val * x_val
                    expected[batch * OUTPUT_NEURONS + i] = acc

        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, MutAnyOrigin](
            x,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, INPUT_NEURONS),
                RuntimeTuple[X_LAYOUT.stride](INPUT_NEURONS, 1)
            )
        )

        dense_layer.allocate_kernel_memory(ctx, BATCH_SIZE)
        output_tensor, output = dense_layer.forward(None, x_tensor)
        ctx.synchronize()

        with output.map_to_host() as output_host:
            for batch in range(BATCH_SIZE):
                for row in range(OUTPUT_NEURONS):
                    idx = batch * OUTPUT_NEURONS + row
                    assert_almost_equal(output_host[idx], expected[idx], rtol=1e-10)

fn test_dense_backward() raises:
    comptime BATCH_SIZE = 32
    comptime INPUT_NEURONS = 128
    comptime OUTPUT_NEURONS = 64
    comptime DTYPE = DType.float64
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime GRAD_OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    computational_graph = ComputationalGraph[DTYPE]()
    computational_graph_ptr = UnsafePointer(to=computational_graph)
    dense_layer = Dense[DTYPE](computational_graph_ptr, INPUT_NEURONS, OUTPUT_NEURONS)
    dense_layer.set_training(True)

    with DeviceContext() as ctx:
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * INPUT_NEURONS)
        x.enqueue_fill(0)
        grad_output = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_NEURONS)
        grad_output.enqueue_fill(0)

        expected_w_gradient = ctx.enqueue_create_host_buffer[DTYPE](OUTPUT_NEURONS * INPUT_NEURONS)
        expected_w_gradient.enqueue_fill(0)
        expected_b_gradient = ctx.enqueue_create_host_buffer[DTYPE](OUTPUT_NEURONS)
        expected_b_gradient.enqueue_fill(0)
        expected_x_gradient = ctx.enqueue_create_host_buffer[DTYPE](BATCH_SIZE * INPUT_NEURONS)
        expected_x_gradient.enqueue_fill(0)

        with x.map_to_host() as x_host, grad_output.map_to_host() as grad_output_host:
            for batch in range(BATCH_SIZE):
                for col in range(INPUT_NEURONS):
                    x_host[batch * INPUT_NEURONS + col] = random_float64()

            for batch in range(BATCH_SIZE):
                for row in range(OUTPUT_NEURONS):
                    grad_output_host[batch * OUTPUT_NEURONS + row] = random_float64()

            w_ptr = dense_layer.w_cpu.unsafe_ptr()

            for i in range(OUTPUT_NEURONS):
                for k in range(INPUT_NEURONS):
                    acc: Float64 = 0.0
                    for batch in range(BATCH_SIZE):
                        acc += grad_output_host[batch * OUTPUT_NEURONS + i] * x_host[batch * INPUT_NEURONS + k]
                    expected_w_gradient[i * INPUT_NEURONS + k] = acc
            
            for i in range(OUTPUT_NEURONS):
                acc: Float64 = 0.0
                for batch in range(BATCH_SIZE):
                    acc += grad_output_host[batch * OUTPUT_NEURONS + i]
                expected_b_gradient[i] = acc
            
            for batch in range(BATCH_SIZE):
                for k in range(INPUT_NEURONS):
                    acc: Float64 = 0.0
                    for i in range(OUTPUT_NEURONS):
                        acc += w_ptr[i * INPUT_NEURONS + k] * grad_output_host[batch * OUTPUT_NEURONS + i]
                    expected_x_gradient[batch * INPUT_NEURONS + k] = acc

        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, MutAnyOrigin](
            x,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, INPUT_NEURONS),
                RuntimeTuple[X_LAYOUT.stride](INPUT_NEURONS, 1)
            )
        )
        grad_output_tensor = LayoutTensor[DTYPE, GRAD_OUTPUT_LAYOUT, MutAnyOrigin](
            grad_output,
            RuntimeLayout[GRAD_OUTPUT_LAYOUT](
                RuntimeTuple[GRAD_OUTPUT_LAYOUT.shape](BATCH_SIZE, OUTPUT_NEURONS),
                RuntimeTuple[GRAD_OUTPUT_LAYOUT.stride](OUTPUT_NEURONS, 1)
            )
        )

        dense_layer.allocate_kernel_memory(ctx, BATCH_SIZE)
        gradients_data = dense_layer.backward(None, x_tensor, grad_output_tensor)
        x_gradient = gradients_data[0][1]
        w_gradient = gradients_data[1][0]
        b_gradient = gradients_data[1][1]
        ctx.synchronize()

        with w_gradient.map_to_host() as w_gradient_host:
            for row in range(OUTPUT_NEURONS):
                for col in range(INPUT_NEURONS):
                    idx = row * INPUT_NEURONS + col
                    assert_almost_equal(w_gradient_host[idx], expected_w_gradient[idx], rtol=1e-10)
        
        with b_gradient.map_to_host() as b_gradient_host:
            for i in range(OUTPUT_NEURONS):
                assert_almost_equal(b_gradient_host[i], expected_b_gradient[i], rtol=1e-10)
        
        with x_gradient.map_to_host() as x_gradient_host:
            for batch in range(BATCH_SIZE):
                for col in range(INPUT_NEURONS):
                    idx = batch * INPUT_NEURONS + col
                    assert_almost_equal(x_gradient_host[idx], expected_x_gradient[idx], rtol=1e-10)

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
