from sys import simd_width_of

from src.layers import Layer

struct SGD[dtype: DType]:
    var learning_rate: Float64

    fn __init__(out self, learning_rate: Float64):
        self.learning_rate = learning_rate

    fn update_weights(self, layer: UnsafePointer[Layer[Self.dtype], MutAnyOrigin], gradients_data: List[List[Float64]]) raises -> None:
        # TO-DO: TBD whether a GPU kernel would be quicker than SIMD for this
        comptime simd_width = simd_width_of[DType.float64]()
        lr_simd = SIMD[DType.float64, simd_width](self.learning_rate)

        scaled_gradients = List[List[Float64]]()
        for i in range(len(gradients_data)):
            num_elements = len(gradients_data[i])
            simd_end = (num_elements // simd_width) * simd_width
            src_ptr = gradients_data[i].unsafe_ptr()

            scaled = List[Float64]()
            scaled.resize(num_elements, 0.0)
            dst_ptr = scaled.unsafe_ptr()

            for j in range(0, simd_end, simd_width):
                grad_vec = src_ptr.load[width=simd_width](j)
                dst_ptr.store[width=simd_width](j, grad_vec * lr_simd)

            for j in range(simd_end, num_elements):
                scaled[j] = gradients_data[i][j] * self.learning_rate

            scaled_gradients.append(scaled^)

        layer[].update_weights(scaled_gradients)
