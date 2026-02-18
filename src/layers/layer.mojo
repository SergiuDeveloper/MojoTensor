from python import PythonObject

from src.computational_graph import ComputationalGraph
from .dense import Dense

comptime Layer = Dense

@fieldwise_init
struct LayerType(Copyable, Movable):
    var value: String

    comptime DENSE = LayerType('Dense')

    fn instantiate[dtype: DType](self, computational_graph: UnsafePointer[ComputationalGraph[dtype], MutAnyOrigin], data: PythonObject) raises -> Layer[dtype]:
        if self.value == LayerType.DENSE.value:
            return Dense[dtype].deserialize(computational_graph, data)
        raise Error('Instantiation of type {} not handled'.format(self.value))
