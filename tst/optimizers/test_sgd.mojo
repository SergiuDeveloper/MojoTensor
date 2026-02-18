from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from testing import TestSuite, assert_almost_equal

from src.layers import Dense
from src.optimizers import Optimizer, SGD
from src.computational_graph import ComputationalGraph

fn test_sgd_update_weights() raises:
    comptime BATCH_SIZE = 2
    comptime INPUT_NEURONS = 4
    comptime OUTPUT_NEURONS = 3
    comptime DTYPE = DType.float64
    comptime LEARNING_RATE = 0.1

    optimizer = SGD[DTYPE](LEARNING_RATE)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr))
    computational_graph_ptr = UnsafePointer(to=computational_graph)
    dense_layer = Dense[DTYPE](computational_graph_ptr, 'dense1', INPUT_NEURONS, OUTPUT_NEURONS)
    dense_layer.set_training(True)

    expected_w = List[Float64]()
    expected_b = List[Float64]()
    w_grad = List[Float64]()
    b_grad = List[Float64]()

    w_ptr = dense_layer.w_cpu.unsafe_ptr()
    b_ptr = dense_layer.b_cpu.unsafe_ptr()

    for i in range(OUTPUT_NEURONS * INPUT_NEURONS):
        grad_val = Float64(i) * 0.01
        w_grad.append(grad_val)
        expected_w.append(w_ptr[i].cast[DType.float64]() - LEARNING_RATE * grad_val)

    for i in range(OUTPUT_NEURONS):
        grad_val = Float64(i) * 0.02
        b_grad.append(grad_val)
        expected_b.append(b_ptr[i].cast[DType.float64]() - LEARNING_RATE * grad_val)

    with DeviceContext() as ctx:
        dense_layer.allocate_kernel_memory(ctx, BATCH_SIZE)

        gradients_data = List[List[Float64]]()
        gradients_data.append(List[Float64]())
        gradients_data.append(List[Float64]())
        for i in range(OUTPUT_NEURONS * INPUT_NEURONS):
            gradients_data[0].append(w_grad[i])
        for i in range(OUTPUT_NEURONS):
            gradients_data[1].append(b_grad[i])

        layer_ptr = UnsafePointer(to=dense_layer)
        optimizer.update_weights(layer_ptr, gradients_data)
        ctx.synchronize()

        w_cpu_ptr = dense_layer.w_cpu.unsafe_ptr()
        for i in range(OUTPUT_NEURONS * INPUT_NEURONS):
            assert_almost_equal(w_cpu_ptr[i].cast[DType.float64](), expected_w[i], rtol=1e-10)

        b_cpu_ptr = dense_layer.b_cpu.unsafe_ptr()
        for i in range(OUTPUT_NEURONS):
            assert_almost_equal(b_cpu_ptr[i].cast[DType.float64](), expected_b[i], rtol=1e-10)

        with dense_layer.w.value().map_to_host() as w_host:
            for i in range(OUTPUT_NEURONS * INPUT_NEURONS):
                assert_almost_equal(w_host[i].cast[DType.float64](), expected_w[i], rtol=1e-10)

        with dense_layer.b.value().map_to_host() as b_host:
            for i in range(OUTPUT_NEURONS):
                assert_almost_equal(b_host[i].cast[DType.float64](), expected_b[i], rtol=1e-10)

        w_tensor_buf = ctx.enqueue_create_host_buffer[DTYPE](OUTPUT_NEURONS * INPUT_NEURONS)
        w_tensor_buf.enqueue_copy_from(dense_layer.w_tensor.value().ptr)
        ctx.synchronize()
        for i in range(OUTPUT_NEURONS * INPUT_NEURONS):
            assert_almost_equal(w_tensor_buf[i].cast[DType.float64](), expected_w[i], rtol=1e-10)

        b_tensor_buf = ctx.enqueue_create_host_buffer[DTYPE](OUTPUT_NEURONS)
        b_tensor_buf.enqueue_copy_from(dense_layer.b_tensor.value().ptr)
        ctx.synchronize()
        for i in range(OUTPUT_NEURONS):
            assert_almost_equal(b_tensor_buf[i].cast[DType.float64](), expected_b[i], rtol=1e-10)

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
