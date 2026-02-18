from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from random import random_float64
from testing import TestSuite, assert_almost_equal, assert_true
from python import Python

from src.layers import Dense
from src.optimizers import Optimizer, SGD
from src.computational_graph import ComputationalGraph

fn test_dense_forward() raises:
    comptime BATCH_SIZE = 32
    comptime INPUT_NEURONS = 128
    comptime OUTPUT_NEURONS = 64
    comptime DTYPE = DType.float64
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime LEARNING_RATE = 0.1

    optimizer = SGD[DTYPE](LEARNING_RATE)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr))
    computational_graph_ptr = UnsafePointer(to=computational_graph)
    dense_layer = Dense[DTYPE](computational_graph_ptr, 'dense1', INPUT_NEURONS, OUTPUT_NEURONS)

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
    comptime LEARNING_RATE = 0.1

    optimizer = SGD[DTYPE](LEARNING_RATE)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr))
    computational_graph_ptr = UnsafePointer(to=computational_graph)
    dense_layer = Dense[DTYPE](computational_graph_ptr, 'dense1', INPUT_NEURONS, OUTPUT_NEURONS)
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

fn test_dense_update_weights() raises:
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
        expected_w.append(w_ptr[i].cast[DType.float64]() - grad_val)

    for i in range(OUTPUT_NEURONS):
        grad_val = Float64(i) * 0.02
        b_grad.append(grad_val)
        expected_b.append(b_ptr[i].cast[DType.float64]() - grad_val)

    with DeviceContext() as ctx:
        dense_layer.allocate_kernel_memory(ctx, BATCH_SIZE)

        gradients_data = List[List[Float64]]()
        gradients_data.append(List[Float64]())
        gradients_data.append(List[Float64]())
        for i in range(OUTPUT_NEURONS * INPUT_NEURONS):
            gradients_data[0].append(w_grad[i])
        for i in range(OUTPUT_NEURONS):
            gradients_data[1].append(b_grad[i])
        dense_layer.update_weights(gradients_data)
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

fn test_dense_serialize() raises:
    json = Python.import_module('json')

    comptime DTYPE = DType.float64
    comptime INPUT_NEURONS = 4
    comptime OUTPUT_NEURONS = 3

    optimizer = SGD[DTYPE](0.1)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr))
    computational_graph_ptr = UnsafePointer(to=computational_graph)

    dense = Dense[DTYPE](computational_graph_ptr, 'dense1', INPUT_NEURONS, OUTPUT_NEURONS)
    serialized = json.loads(json.dumps(dense.serialize()))

    assert_true(serialized['metadata']['input_neurons'] == dense.input_neurons)
    assert_true(serialized['metadata']['output_neurons'] == dense.output_neurons)

    w_ptr = dense.w_cpu.unsafe_ptr()
    for i in range(OUTPUT_NEURONS):
        for j in range(INPUT_NEURONS):
            assert_true(serialized['weights']['w'][i][j] == w_ptr[i * INPUT_NEURONS + j].cast[DType.float64]())

    b_ptr = dense.b_cpu.unsafe_ptr()
    for i in range(OUTPUT_NEURONS):
        assert_true(serialized['weights']['b'][i] == b_ptr[i].cast[DType.float64]())

fn test_dense_deserialize() raises:
    comptime DTYPE = DType.float64
    comptime INPUT_NEURONS = 4
    comptime OUTPUT_NEURONS = 3

    optimizer = SGD[DTYPE](0.1)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr))
    computational_graph_ptr = UnsafePointer(to=computational_graph)

    original = Dense[DTYPE](computational_graph_ptr, 'test_dense', INPUT_NEURONS, OUTPUT_NEURONS)

    entry = Python.evaluate('{}')
    entry['name'] = original.name
    entry['data'] = original.serialize()

    deserialized = Dense[DTYPE].deserialize(computational_graph_ptr, entry)

    assert_true(deserialized.name == original.name)
    assert_true(deserialized.LAYER_TYPE.value == original.LAYER_TYPE.value)
    assert_true(deserialized.input_neurons == original.input_neurons)
    assert_true(deserialized.output_neurons == original.output_neurons)

    orig_w_ptr = original.w_cpu.unsafe_ptr()
    deser_w_ptr = deserialized.w_cpu.unsafe_ptr()
    for i in range(OUTPUT_NEURONS):
        for j in range(INPUT_NEURONS):
            assert_almost_equal(
                deser_w_ptr[i * INPUT_NEURONS + j].cast[DType.float64](),
                orig_w_ptr[i * INPUT_NEURONS + j].cast[DType.float64](),
                rtol=1e-10
            )

    orig_b_ptr = original.b_cpu.unsafe_ptr()
    deser_b_ptr = deserialized.b_cpu.unsafe_ptr()
    for i in range(OUTPUT_NEURONS):
        assert_almost_equal(
            deser_b_ptr[i].cast[DType.float64](),
            orig_b_ptr[i].cast[DType.float64](),
            rtol=1e-10
        )

fn test_dense_set_weights() raises:
    comptime DTYPE = DType.float64
    comptime INPUT_NEURONS = 4
    comptime OUTPUT_NEURONS = 3

    optimizer = SGD[DTYPE](0.1)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr))
    computational_graph_ptr = UnsafePointer(to=computational_graph)

    original = Dense[DTYPE](computational_graph_ptr, 'test_dense', INPUT_NEURONS, OUTPUT_NEURONS)
    target = Dense[DTYPE](computational_graph_ptr, 'other_dense', INPUT_NEURONS, OUTPUT_NEURONS)
    target.set_weights(original.serialize()['weights'])

    orig_w_ptr = original.w_cpu.unsafe_ptr()
    target_w_ptr = target.w_cpu.unsafe_ptr()
    for i in range(OUTPUT_NEURONS):
        for j in range(INPUT_NEURONS):
            assert_almost_equal(
                target_w_ptr[i * INPUT_NEURONS + j].cast[DType.float64](),
                orig_w_ptr[i * INPUT_NEURONS + j].cast[DType.float64](),
                rtol=1e-10
            )

    orig_b_ptr = original.b_cpu.unsafe_ptr()
    target_b_ptr = target.b_cpu.unsafe_ptr()
    for i in range(OUTPUT_NEURONS):
        assert_almost_equal(
            target_b_ptr[i].cast[DType.float64](),
            orig_b_ptr[i].cast[DType.float64](),
            rtol=1e-10
        )

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
