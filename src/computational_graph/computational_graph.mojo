from utils import Variant
from layout import Layout, LayoutTensor
from layout.int_tuple import UNKNOWN_VALUE
from sys import simd_width_of

from src.layers import Layer, LayerFuncTypeConstants

struct ComputationalGraph[dtype: DType](ImplicitlyCopyable):
    var backward_operation_inputs: List[Tuple[UnsafePointer[Layer[Self.dtype], MutAnyOrigin], Optional[UnsafePointer[Layer[Self.dtype], MutAnyOrigin]], LayerFuncTypeConstants[Self.dtype].LayerInputType]]
    var backward_operation_grad_outputs: Dict[String, List[LayerFuncTypeConstants[Self.dtype].LayerGradOutputType]]
    var gradients: Dict[String, List[List[Float64]]]
    var layer_keys: Dict[String, UnsafePointer[Layer[Self.dtype], MutAnyOrigin]]

    fn __init__(out self):
        self.backward_operation_inputs = []
        self.backward_operation_grad_outputs = {}
        self.gradients = {}
        self.layer_keys = {}

    fn __copyinit__(out self, existing: Self):
        self.backward_operation_inputs = List[Tuple[UnsafePointer[Layer[Self.dtype], MutAnyOrigin], Optional[UnsafePointer[Layer[Self.dtype], MutAnyOrigin]], LayerFuncTypeConstants[Self.dtype].LayerInputType]]()
        for item in existing.backward_operation_inputs:
            self.backward_operation_inputs.append(item)
        
        self.backward_operation_grad_outputs = {}
        for entry in existing.backward_operation_grad_outputs.items():
            key1 = entry.key
            copied_list1: List[LayerFuncTypeConstants[Self.dtype].LayerGradOutputType] = []
            for grad in entry.value:
                copied_list1.append(grad)
            self.backward_operation_grad_outputs[key1] = copied_list1^

        self.gradients = {}
        for entry in existing.gradients.items():
            key2 = entry.key
            copied_list2: List[List[Float64]] = []
            try:
                for grad_list in existing.gradients[key2]:
                    copied_inner_list = List[Float64]()
                    for elem in grad_list:
                        copied_inner_list.append(elem)
                    copied_list2.append(copied_inner_list^)
            except:
                pass
            self.gradients[key2] = copied_list2^

        self.layer_keys = {}
        for entry in existing.layer_keys.items():
            self.layer_keys[entry.key] = entry.value

    fn add_backward_operation_inputs(
        mut self,
        current_layer: UnsafePointer[Layer[Self.dtype], MutAnyOrigin],
        previous_layer: Optional[UnsafePointer[Layer[Self.dtype], MutAnyOrigin]],
        input_tensor: LayerFuncTypeConstants[Self.dtype].LayerInputType
    ) -> None:
        self.backward_operation_inputs.append(Tuple(current_layer, previous_layer, input_tensor))

    fn add_backward_operation_grad_outputs(
        mut self,
        previous_layer: Optional[UnsafePointer[Layer[Self.dtype], MutAnyOrigin]],
        grad_output_tensor: LayerFuncTypeConstants[Self.dtype].LayerGradOutputType
    ) raises -> None:
        if previous_layer is None:
            return
        layer_key = self._compute_layer_key(previous_layer.value())
        if layer_key not in self.backward_operation_grad_outputs:
            self.backward_operation_grad_outputs[layer_key] = [grad_output_tensor]
        else:
            self.backward_operation_grad_outputs[layer_key].append(grad_output_tensor)

    fn backward(mut self, grad_output: LayerFuncTypeConstants[Self.dtype].LayerGradOutputType) raises -> None:
        first_iteration = True
        while len(self.backward_operation_inputs) > 0:
            backward_operation_input = self.backward_operation_inputs.pop()
            current_layer, previous_layer, input_tensor = backward_operation_input
            current_grad_output = grad_output if first_iteration else self._retrieve_latest_grad_output(current_layer)
            first_iteration = False
            
            gradient_data = current_layer[].backward(previous_layer, input_tensor, current_grad_output)
            x_gradient_data = gradient_data[0]
            param_gradients_data = gradient_data[1].copy()

            current_layer_key = self._compute_layer_key(current_layer)
            if current_layer_key not in self.gradients:
                self.gradients[current_layer_key] = {}
                self.layer_keys[current_layer_key] = current_layer

            if len(self.gradients[current_layer_key]) == 0:
                cpu_gradients = List[List[Float64]]()
                for i in range(len(param_gradients_data)):
                    cpu_list = List[Float64]()
                    with param_gradients_data[i].map_to_host() as gpu_data:
                        for j in range(len(gpu_data)):
                            cpu_list.append(gpu_data[j].cast[DType.float64]())
                    cpu_gradients.append(cpu_list^)
                self.gradients[current_layer_key] = cpu_gradients^
            else:
                for i in range(len(self.gradients[current_layer_key])):
                    # TO-DO: TBD whether a GPU kernel would be quicker than SIMD for this
                    with param_gradients_data[i].map_to_host() as param_grad_host:
                        comptime simd_width = simd_width_of[Self.dtype]()
                        num_elements = len(self.gradients[current_layer_key][i])
                        simd_end = (num_elements // simd_width) * simd_width
                        
                        # SIMD add
                        for j in range(0, simd_end, simd_width):
                            grad_simd = self.gradients[current_layer_key][i].unsafe_ptr().load[width=simd_width](j)
                            param_simd = param_grad_host.unsafe_ptr().load[width=simd_width](j).cast[DType.float64]()
                            self.gradients[current_layer_key][i].unsafe_ptr().store[width=simd_width](j, grad_simd + param_simd)
                        
                        # Handle remainder
                        for j in range(simd_end, num_elements):
                            self.gradients[current_layer_key][i][j] += param_grad_host[j].cast[DType.float64]()

            self.add_backward_operation_grad_outputs(previous_layer, x_gradient_data[0].copy())

    fn update_weights(mut self) -> None:
        for layer_key, gradients_data in self.gradients.items():
            layer = self.layer_keys[layer_key]
            # TO-DO: Update based on optimizer rules

        self.gradients = {}
        self.layer_keys = {}

    fn _retrieve_latest_grad_output(mut self, layer: UnsafePointer[Layer[Self.dtype], MutAnyOrigin]) raises -> LayerFuncTypeConstants[Self.dtype].LayerGradOutputType:
        layer_key = self._compute_layer_key(layer)
        return self.backward_operation_grad_outputs[layer_key].pop()

    fn _compute_layer_key(self, layer: UnsafePointer[Layer[Self.dtype], MutAnyOrigin]) -> String:
        return layer.__str__()
