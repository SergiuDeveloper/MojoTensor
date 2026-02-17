from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from random import random_float64
from testing import TestSuite, assert_equal, assert_true, assert_almost_equal

from src.layers import Dense, LayerFuncTypeConstants
from src.computational_graph import ComputationalGraph

fn test_computational_graph_backward() raises:
    comptime BATCH_SIZE = 16
    comptime INPUT_NEURONS = 64
    comptime HIDDEN_NEURONS = 32
    comptime OUTPUT_NEURONS = 16
    comptime DTYPE = DType.float64
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime GRAD_OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    computational_graph = ComputationalGraph[DTYPE]()
    computational_graph_ptr = UnsafePointer(to=computational_graph)
    layer1 = Dense[DTYPE](computational_graph_ptr, INPUT_NEURONS, HIDDEN_NEURONS)
    layer2 = Dense[DTYPE](computational_graph_ptr, HIDDEN_NEURONS, OUTPUT_NEURONS)
    layer1.set_training(True)
    layer2.set_training(True)

    with DeviceContext() as ctx:
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * INPUT_NEURONS)
        grad_output = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_NEURONS)

        with x.map_to_host() as x_host:
            for i in range(BATCH_SIZE * INPUT_NEURONS):
                x_host[i] = random_float64()

        with grad_output.map_to_host() as grad_output_host:
            for i in range(BATCH_SIZE * OUTPUT_NEURONS):
                grad_output_host[i] = random_float64()

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

        layer1.allocate_kernel_memory(ctx, BATCH_SIZE)
        layer2.allocate_kernel_memory(ctx, BATCH_SIZE)

        layer1.set_training(False)
        layer2.set_training(False)
        hidden_tensor_ref, hidden_buffer_ref = layer1.forward(None, x_tensor)
        ctx.synchronize()
        layer1.set_training(True)
        layer2.set_training(True)

        expected_layer2_grads = layer2.backward(None, hidden_tensor_ref, grad_output_tensor)
        expected_x2_grad_buffer = expected_layer2_grads[0][1]
        expected_w2_grad = expected_layer2_grads[1][0]
        expected_b2_grad = expected_layer2_grads[1][1]

        x2_grad_tensor = LayoutTensor[DTYPE, X_LAYOUT, MutAnyOrigin](
            expected_x2_grad_buffer,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, HIDDEN_NEURONS),
                RuntimeTuple[X_LAYOUT.stride](HIDDEN_NEURONS, 1)
            )
        )

        expected_layer1_grads = layer1.backward(None, x_tensor, x2_grad_tensor)
        expected_w1_grad = expected_layer1_grads[1][0]
        expected_b1_grad = expected_layer1_grads[1][1]
        ctx.synchronize()

        layer1.w_gradient.value().enqueue_fill(0)
        layer1.b_gradient.value().enqueue_fill(0)
        layer1.x_gradient.value().enqueue_fill(0)
        layer2.w_gradient.value().enqueue_fill(0)
        layer2.b_gradient.value().enqueue_fill(0)
        layer2.x_gradient.value().enqueue_fill(0)

        hidden_tensor, hidden_buffer = layer1.forward(None, x_tensor)
        output_tensor, output_buffer = layer2.forward(UnsafePointer[Dense[DTYPE], MutAnyOrigin](UnsafePointer(to=layer1)), hidden_tensor)
        ctx.synchronize()

        assert_equal(len(computational_graph.backward_operation_inputs), 2)

        grad_output_variant = LayerFuncTypeConstants[DTYPE].LayerGradOutputType(grad_output_tensor)
        computational_graph.backward(grad_output_variant)
        ctx.synchronize()

        layer1_key = computational_graph._compute_layer_key(UnsafePointer(to=layer1))
        layer2_key = computational_graph._compute_layer_key(UnsafePointer(to=layer2))

        assert_true(layer1_key in computational_graph.gradients)
        assert_true(layer2_key in computational_graph.gradients)
        assert_true(layer1_key in computational_graph.layer_keys)
        assert_true(layer2_key in computational_graph.layer_keys)

        layer1_grads = List[List[Float64]]()
        for i in range(len(computational_graph.gradients[layer1_key])):
            grad_copy = List[Float64]()
            for j in range(len(computational_graph.gradients[layer1_key][i])):
                grad_copy.append(computational_graph.gradients[layer1_key][i][j])
            layer1_grads.append(grad_copy^)

        layer2_grads = List[List[Float64]]()
        for i in range(len(computational_graph.gradients[layer2_key])):
            grad_copy = List[Float64]()
            for j in range(len(computational_graph.gradients[layer2_key][i])):
                grad_copy.append(computational_graph.gradients[layer2_key][i][j])
            layer2_grads.append(grad_copy^)

        assert_equal(len(layer1_grads), 2)
        assert_equal(len(layer2_grads), 2)

        assert_equal(len(layer2_grads[0]), OUTPUT_NEURONS * HIDDEN_NEURONS)
        with expected_w2_grad.map_to_host() as expected_w2_host:
            for i in range(OUTPUT_NEURONS * HIDDEN_NEURONS):
                assert_almost_equal(layer2_grads[0][i], expected_w2_host[i].cast[DType.float64](), rtol=1e-10)

        assert_equal(len(layer2_grads[1]), OUTPUT_NEURONS)
        with expected_b2_grad.map_to_host() as expected_b2_host:
            for i in range(OUTPUT_NEURONS):
                assert_almost_equal(layer2_grads[1][i], expected_b2_host[i].cast[DType.float64](), rtol=1e-10)

        assert_equal(len(layer1_grads[0]), HIDDEN_NEURONS * INPUT_NEURONS)
        with expected_w1_grad.map_to_host() as expected_w1_host:
            for i in range(HIDDEN_NEURONS * INPUT_NEURONS):
                assert_almost_equal(layer1_grads[0][i], expected_w1_host[i].cast[DType.float64](), rtol=1e-10)

        assert_equal(len(layer1_grads[1]), HIDDEN_NEURONS)
        with expected_b1_grad.map_to_host() as expected_b1_host:
            for i in range(HIDDEN_NEURONS):
                assert_almost_equal(layer1_grads[1][i], expected_b1_host[i].cast[DType.float64](), rtol=1e-10)

        assert_equal(len(computational_graph.backward_operation_inputs), 0)

        # TO-DO: Add assertion for update_weights

fn test_computational_graph_copy() raises:
    comptime BATCH_SIZE = 8
    comptime INPUT_NEURONS = 32
    comptime OUTPUT_NEURONS = 16
    comptime DTYPE = DType.float64
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime GRAD_OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    original_graph = ComputationalGraph[DTYPE]()
    computational_graph_ptr = UnsafePointer(to=original_graph)
    layer1 = Dense[DTYPE](computational_graph_ptr, INPUT_NEURONS, OUTPUT_NEURONS)
    layer1.set_training(True)

    with DeviceContext() as ctx:
        x = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * INPUT_NEURONS)
        grad_output = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_NEURONS)

        with x.map_to_host() as x_host:
            for i in range(BATCH_SIZE * INPUT_NEURONS):
                x_host[i] = random_float64()

        with grad_output.map_to_host() as grad_output_host:
            for i in range(BATCH_SIZE * OUTPUT_NEURONS):
                grad_output_host[i] = random_float64()

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

        layer1.allocate_kernel_memory(ctx, BATCH_SIZE)

        layer1.set_training(False)
        _ = layer1.forward(None, x_tensor)
        ctx.synchronize()
        layer1.set_training(True)

        expected_grads = layer1.backward(None, x_tensor, grad_output_tensor)
        expected_w_grad = expected_grads[1][0]
        expected_b_grad = expected_grads[1][1]
        ctx.synchronize()

        layer1.w_gradient.value().enqueue_fill(0)
        layer1.b_gradient.value().enqueue_fill(0)
        layer1.x_gradient.value().enqueue_fill(0)

        output_tensor, output_buffer = layer1.forward(None, x_tensor)
        ctx.synchronize()

        assert_equal(len(original_graph.backward_operation_inputs), 1)

        copied_graph = original_graph

        assert_equal(len(copied_graph.backward_operation_inputs), 1)

        grad_output_variant = LayerFuncTypeConstants[DTYPE].LayerGradOutputType(grad_output_tensor)
        original_graph.backward(grad_output_variant)
        ctx.synchronize()

        assert_equal(len(original_graph.backward_operation_inputs), 0)
        assert_equal(len(copied_graph.backward_operation_inputs), 1)

        layer1_key = original_graph._compute_layer_key(UnsafePointer(to=layer1))

        assert_true(layer1_key in original_graph.gradients)
        assert_true(layer1_key in original_graph.layer_keys)
        assert_true(layer1_key not in copied_graph.gradients)
        assert_true(layer1_key not in copied_graph.layer_keys)

        original_grads = List[List[Float64]]()
        for i in range(len(original_graph.gradients[layer1_key])):
            grad_copy = List[Float64]()
            for j in range(len(original_graph.gradients[layer1_key][i])):
                grad_copy.append(original_graph.gradients[layer1_key][i][j])
            original_grads.append(grad_copy^)

        assert_equal(len(original_grads), 2)

        assert_equal(len(original_grads[0]), OUTPUT_NEURONS * INPUT_NEURONS)
        with expected_w_grad.map_to_host() as expected_w_host:
            for i in range(OUTPUT_NEURONS * INPUT_NEURONS):
                assert_almost_equal(original_grads[0][i], expected_w_host[i].cast[DType.float64](), rtol=1e-10)

        assert_equal(len(original_grads[1]), OUTPUT_NEURONS)
        with expected_b_grad.map_to_host() as expected_b_host:
            for i in range(OUTPUT_NEURONS):
                assert_almost_equal(original_grads[1][i], expected_b_host[i].cast[DType.float64](), rtol=1e-10)

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
