from python import Python

from src.computational_graph import ComputationalGraph
from .layer import Layer, LayerType

fn save_layers[dtype: DType](layers: List[UnsafePointer[Layer[dtype], MutAnyOrigin]], file_path: String) raises -> None:
    pickle = Python.import_module('pickle')
    builtins = Python.import_module('builtins')

    layer_data = Python.evaluate('[]')
    for layer in layers:
        entry = Python.evaluate('{}')
        entry['type'] = layer[].LAYER_TYPE.value
        entry['name'] = layer[].name
        entry['data'] = layer[].serialize()
        layer_data.append(entry)

    file = builtins.open(file_path, 'wb')
    try:
        pickle.dump(layer_data, file)
    finally:
        file.close()

fn load_layers[dtype: DType](computational_graph: UnsafePointer[ComputationalGraph[dtype], MutAnyOrigin], file_path: String) raises -> List[Layer[dtype]]:
    pickle = Python.import_module('pickle')
    builtins = Python.import_module('builtins')

    file = builtins.open(file_path, 'rb')
    try:
        layers_data = pickle.load(file)
    finally:
        file.close()

    layers = List[Layer[dtype]]()
    for i in range(len(layers_data)):
        layers.append(LayerType(String(layers_data[i]['type'])).instantiate[dtype](computational_graph, layers_data[i]))
    return layers^
