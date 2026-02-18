import os
from testing import TestSuite, assert_true, assert_almost_equal
from python import Python

from src.layers import save_layers, load_layers, Dense, Layer
from src.computational_graph import ComputationalGraph
from src.optimizers import Optimizer, SGD

fn test_save_layers() raises:
    pickle = Python.import_module('pickle')
    builtins = Python.import_module('builtins')

    comptime DTYPE = DType.float64
    comptime TEMP_PATH = './temp_weights.pkl'

    optimizer = SGD[DTYPE](0.1)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr))
    computational_graph_ptr = UnsafePointer(to=computational_graph)

    dense1 = Dense[DTYPE](computational_graph_ptr, 'dense1', 4, 3)
    dense2 = Dense[DTYPE](computational_graph_ptr, 'dense2', 3, 2)
    dense3 = Dense[DTYPE](computational_graph_ptr, 'dense3', 2, 1)

    layers = List[UnsafePointer[Layer[DTYPE], MutAnyOrigin]]()
    layers.append(UnsafePointer(to=dense1))
    layers.append(UnsafePointer(to=dense2))
    layers.append(UnsafePointer(to=dense3))

    try:
        save_layers[DTYPE](layers, TEMP_PATH)

        file = builtins.open(TEMP_PATH, 'rb')
        parsed = pickle.load(file)
        file.close()

        assert_true(parsed[0]['type'] == dense1.LAYER_TYPE.value)
        assert_true(parsed[0]['name'] == dense1.name)
        assert_true(parsed[0]['data'] == dense1.serialize())

        assert_true(parsed[1]['type'] == dense1.LAYER_TYPE.value)
        assert_true(parsed[1]['name'] == dense2.name)
        assert_true(parsed[1]['data'] == dense2.serialize())

        assert_true(parsed[2]['type'] == dense1.LAYER_TYPE.value)
        assert_true(parsed[2]['name'] == dense3.name)
        assert_true(parsed[2]['data'] == dense3.serialize())
    finally:
        if os.path.exists(TEMP_PATH):
            os.remove(TEMP_PATH)

fn test_load_layers() raises:
    comptime DTYPE = DType.float64
    comptime TEMP_PATH = './temp_weights_load.pkl'

    optimizer = SGD[DTYPE](0.1)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr))
    computational_graph_ptr = UnsafePointer(to=computational_graph)

    dense1 = Dense[DTYPE](computational_graph_ptr, 'dense1', 4, 3)
    dense2 = Dense[DTYPE](computational_graph_ptr, 'dense2', 3, 2)

    layers = List[UnsafePointer[Layer[DTYPE], MutAnyOrigin]]()
    layers.append(UnsafePointer(to=dense1))
    layers.append(UnsafePointer(to=dense2))

    try:
        save_layers[DTYPE](layers, TEMP_PATH)
        loaded = load_layers[DTYPE](computational_graph_ptr, TEMP_PATH)

        assert_true(loaded[0].name == dense1.name)
        assert_true(loaded[0].input_neurons == dense1.input_neurons)
        assert_true(loaded[0].output_neurons == dense1.output_neurons)

        assert_true(loaded[1].name == dense2.name)
        assert_true(loaded[1].input_neurons == dense2.input_neurons)
        assert_true(loaded[1].output_neurons == dense2.output_neurons)

        d1_w_ptr = dense1.w_cpu.unsafe_ptr()
        l0_w_ptr = loaded[0].w_cpu.unsafe_ptr()
        for i in range(dense1.output_neurons):
            for j in range(dense1.input_neurons):
                assert_almost_equal(
                    l0_w_ptr[i * dense1.input_neurons + j].cast[DType.float64](),
                    d1_w_ptr[i * dense1.input_neurons + j].cast[DType.float64](),
                    rtol=1e-10
                )

        d1_b_ptr = dense1.b_cpu.unsafe_ptr()
        l0_b_ptr = loaded[0].b_cpu.unsafe_ptr()
        for i in range(dense1.output_neurons):
            assert_almost_equal(
                l0_b_ptr[i].cast[DType.float64](),
                d1_b_ptr[i].cast[DType.float64](),
                rtol=1e-10
            )

        d2_w_ptr = dense2.w_cpu.unsafe_ptr()
        l1_w_ptr = loaded[1].w_cpu.unsafe_ptr()
        for i in range(dense2.output_neurons):
            for j in range(dense2.input_neurons):
                assert_almost_equal(
                    l1_w_ptr[i * dense2.input_neurons + j].cast[DType.float64](),
                    d2_w_ptr[i * dense2.input_neurons + j].cast[DType.float64](),
                    rtol=1e-10
                )

        d2_b_ptr = dense2.b_cpu.unsafe_ptr()
        l1_b_ptr = loaded[1].b_cpu.unsafe_ptr()
        for i in range(dense2.output_neurons):
            assert_almost_equal(
                l1_b_ptr[i].cast[DType.float64](),
                d2_b_ptr[i].cast[DType.float64](),
                rtol=1e-10
            )
    finally:
        if os.path.exists(TEMP_PATH):
            os.remove(TEMP_PATH)

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
