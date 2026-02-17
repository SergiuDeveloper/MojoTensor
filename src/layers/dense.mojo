from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from gpu.host import DeviceContext, DeviceBuffer
from random import rand

from src.computational_graph import ComputationalGraph
from src.kernels.layers import dense_forward, dense_backward
from src.kernels.constants import MAX_GRID_SIZE
from src.layers.layer import Layer
from src.layers.constants import LayerFuncTypeConstants

struct Dense[dtype: DType]:
    comptime TPB = 16
    comptime OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime W_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime B_LAYOUT = Layout.row_major(UNKNOWN_VALUE)
    comptime X_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime W_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime B_GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE)
    comptime GRAD_OUTPUT_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime FORWARD_KERNEL = dense_forward[
        Self.TPB,
        Self.dtype,
        Self.OUTPUT_LAYOUT,
        Self.X_LAYOUT,
        Self.W_LAYOUT,
        Self.B_LAYOUT
    ]
    comptime BACKWARD_KERNEL = dense_backward[
        Self.TPB,
        Self.dtype,
        Self.X_GRAD_LAYOUT,
        Self.W_GRAD_LAYOUT,
        Self.B_GRAD_LAYOUT,
        Self.X_LAYOUT,
        Self.W_LAYOUT,
        Self.GRAD_OUTPUT_LAYOUT
    ]

    var computational_graph: UnsafePointer[ComputationalGraph[Self.dtype], MutAnyOrigin]
    var input_neurons: Int
    var output_neurons: Int
    var training: Bool
    var forward_memory_allocated: Bool
    var backward_memory_allocated: Bool
    var cpu_ctx: DeviceContext
    var w_cpu: DeviceBuffer[Self.dtype]
    var b_cpu: DeviceBuffer[Self.dtype]
    var ctx: Optional[DeviceContext]
    var batch_size: Optional[Int]
    var w: Optional[DeviceBuffer[Self.dtype]]
    var b: Optional[DeviceBuffer[Self.dtype]]
    var output: Optional[DeviceBuffer[Self.dtype]]
    var output_tensor: Optional[LayoutTensor[Self.dtype, Self.OUTPUT_LAYOUT, MutAnyOrigin]]
    var x_gradient: Optional[DeviceBuffer[Self.dtype]]
    var x_gradient_tensor: Optional[LayoutTensor[Self.dtype, Self.X_GRAD_LAYOUT, MutAnyOrigin]]
    var w_gradient: Optional[DeviceBuffer[Self.dtype]]
    var w_gradient_tensor: Optional[LayoutTensor[Self.dtype, Self.W_GRAD_LAYOUT, MutAnyOrigin]]
    var b_gradient: Optional[DeviceBuffer[Self.dtype]]
    var b_gradient_tensor: Optional[LayoutTensor[Self.dtype, Self.B_GRAD_LAYOUT, MutAnyOrigin]]
    var w_tensor: Optional[LayoutTensor[Self.dtype, Self.W_LAYOUT, ImmutAnyOrigin]]
    var b_tensor: Optional[LayoutTensor[Self.dtype, Self.B_LAYOUT, ImmutAnyOrigin]]
    var forward_grid_dim: Optional[Tuple[Int, Int, Int]]
    var forward_block_dim: Optional[Tuple[Int, Int]]
    var backward_grid_dim: Optional[Tuple[Int, Int, Int]]
    var backward_block_dim: Optional[Tuple[Int, Int, Int]]

    fn __init__(out self, computational_graph: UnsafePointer[ComputationalGraph[Self.dtype], MutAnyOrigin], input_neurons: Int, output_neurons: Int) raises:
        self.computational_graph = computational_graph
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.training = False
        self.forward_memory_allocated = False
        self.backward_memory_allocated = False
        self.cpu_ctx = DeviceContext(api='cpu')

        self.w_cpu = self.cpu_ctx.enqueue_create_buffer[Self.dtype](self.output_neurons * self.input_neurons)
        self.b_cpu = self.cpu_ctx.enqueue_create_buffer[Self.dtype](self.output_neurons)
        rand[Self.dtype](self.w_cpu.unsafe_ptr(), self.output_neurons * self.input_neurons)
        rand[Self.dtype](self.b_cpu.unsafe_ptr(), self.output_neurons)

        self.ctx = None
        self.batch_size = None
        self.w = None
        self.b = None
        self.output = None
        self.output_tensor = None
        self.x_gradient = None
        self.x_gradient_tensor = None
        self.w_gradient = None
        self.w_gradient_tensor = None
        self.b_gradient = None
        self.b_gradient_tensor = None
        self.w_tensor = None
        self.b_tensor = None
        self.forward_grid_dim = None
        self.forward_block_dim = None
        self.backward_grid_dim = None
        self.backward_block_dim = None

    fn set_training(mut self, training: Bool) -> None:
        self.training = training

    fn allocate_kernel_memory(mut self, ctx: DeviceContext, batch_size: Int) raises -> None:
        self.forward_memory_allocated = False
        self.backward_memory_allocated = False

        self._allocate_common_kernel_memory(ctx, batch_size)
        self._allocate_forward_kernel_memory()
        if self.training:
            self._allocate_backward_kernel_memory()

    fn _allocate_common_kernel_memory(mut self, ctx: DeviceContext, batch_size: Int) raises -> None:
        self.ctx = ctx
        self.batch_size = batch_size

        self.w = self.ctx.value().enqueue_create_buffer[Self.dtype](self.output_neurons * self.input_neurons)
        self.w.value().enqueue_copy_from(self.w_cpu)
        self.b = self.ctx.value().enqueue_create_buffer[Self.dtype](self.output_neurons)
        self.b.value().enqueue_copy_from(self.b_cpu)

        self.w_tensor = LayoutTensor[Self.dtype, Self.W_LAYOUT, ImmutAnyOrigin](
            self.w.value(),
            RuntimeLayout[Self.W_LAYOUT](
                RuntimeTuple[Self.W_LAYOUT.shape](self.output_neurons, self.input_neurons),
                RuntimeTuple[Self.W_LAYOUT.stride](self.input_neurons, 1)
            )
        )
        self.b_tensor = LayoutTensor[Self.dtype, Self.B_LAYOUT, ImmutAnyOrigin](
            self.b.value(),
            RuntimeLayout[Self.B_LAYOUT](
                RuntimeTuple[Self.B_LAYOUT.shape](self.output_neurons),
                RuntimeTuple[Self.B_LAYOUT.stride](1)
            )
        )

    fn _allocate_forward_kernel_memory(mut self) raises -> None:
        self.output = self.ctx.value().enqueue_create_buffer[Self.dtype](self.batch_size.value() * self.output_neurons)
        self.output.value().enqueue_fill(0)

        self.output_tensor = LayoutTensor[Self.dtype, Self.OUTPUT_LAYOUT, MutAnyOrigin](
            self.output.value(),
            RuntimeLayout[Self.OUTPUT_LAYOUT](
                RuntimeTuple[Self.OUTPUT_LAYOUT.shape](self.batch_size.value(), self.output_neurons),
                RuntimeTuple[Self.OUTPUT_LAYOUT.stride](self.output_neurons, 1)
            )
        )

        output_blocks = (self.output_neurons + Self.TPB - 1) // Self.TPB
        batch_blocks = (self.batch_size.value() + Self.TPB - 1) // Self.TPB
        total_blocks = output_blocks * batch_blocks

        grid_x = min(total_blocks, MAX_GRID_SIZE)
        grid_y = (total_blocks + grid_x - 1) // grid_x
        grid_z = 1

        self.forward_grid_dim = (grid_x, grid_y, grid_z)
        self.forward_block_dim = (Self.TPB, Self.TPB)

        self.forward_memory_allocated = True

    fn _allocate_backward_kernel_memory(mut self) raises -> None:
        self.x_gradient = self.ctx.value().enqueue_create_buffer[Self.dtype](self.batch_size.value() * self.input_neurons)
        self.x_gradient.value().enqueue_fill(0)
        self.w_gradient = self.ctx.value().enqueue_create_buffer[Self.dtype](self.output_neurons * self.input_neurons)
        self.w_gradient.value().enqueue_fill(0)
        self.b_gradient = self.ctx.value().enqueue_create_buffer[Self.dtype](self.output_neurons)
        self.b_gradient.value().enqueue_fill(0)

        self.x_gradient_tensor = LayoutTensor[Self.dtype, Self.X_GRAD_LAYOUT, MutAnyOrigin](
            self.x_gradient.value(),
            RuntimeLayout[Self.X_GRAD_LAYOUT](
                RuntimeTuple[Self.X_GRAD_LAYOUT.shape](self.batch_size.value(), self.input_neurons),
                RuntimeTuple[Self.X_GRAD_LAYOUT.stride](self.input_neurons, 1)
            )
        )
        self.w_gradient_tensor = LayoutTensor[Self.dtype, Self.W_GRAD_LAYOUT, MutAnyOrigin](
            self.w_gradient.value(),
            RuntimeLayout[Self.W_GRAD_LAYOUT](
                RuntimeTuple[Self.W_GRAD_LAYOUT.shape](self.output_neurons, self.input_neurons),
                RuntimeTuple[Self.W_GRAD_LAYOUT.stride](self.input_neurons, 1)
            )
        )
        self.b_gradient_tensor = LayoutTensor[Self.dtype, Self.B_GRAD_LAYOUT, MutAnyOrigin](
            self.b_gradient.value(),
            RuntimeLayout[Self.B_GRAD_LAYOUT](
                RuntimeTuple[Self.B_GRAD_LAYOUT.shape](self.output_neurons),
                RuntimeTuple[Self.B_GRAD_LAYOUT.stride](1)
            )
        )

        w_blocks = ((self.output_neurons + Self.TPB - 1) // Self.TPB) * ((self.input_neurons + Self.TPB - 1) // Self.TPB)
        b_blocks = (self.output_neurons + Self.TPB - 1) // Self.TPB
        x_blocks = ((self.batch_size.value() + Self.TPB - 1) // Self.TPB) * ((self.input_neurons + Self.TPB - 1) // Self.TPB)

        total_blocks = w_blocks + b_blocks + x_blocks

        grid_x = min(total_blocks, MAX_GRID_SIZE)
        grid_y = (total_blocks + grid_x - 1) // grid_x
        grid_z = 1

        self.backward_grid_dim = (grid_x, grid_y, grid_z)
        self.backward_block_dim = (Self.TPB, Self.TPB, 1)

        self.backward_memory_allocated = True

    fn forward(
        mut self,
        previous_layer: Optional[UnsafePointer[Layer[Self.dtype], MutAnyOrigin]],
        x_tensor_raw: LayerFuncTypeConstants[Self.dtype].LayerInputType
    ) raises -> Tuple[
        LayoutTensor[Self.dtype, Self.OUTPUT_LAYOUT, MutAnyOrigin],
        DeviceBuffer[Self.dtype]
    ]:
        x_tensor = x_tensor_raw[LayoutTensor[Self.dtype, Self.X_LAYOUT, MutAnyOrigin]]

        if not self.forward_memory_allocated:
            raise Error('Kernel memory must be allocated before calling the forward method')

        self.ctx.value().enqueue_function[Self.FORWARD_KERNEL, Self.FORWARD_KERNEL](
            self.output_tensor.value(),
            x_tensor,
            self.w_tensor.value(),
            self.b_tensor.value(),
            self.batch_size.value(),
            self.output_neurons,
            self.input_neurons,
            grid_dim=self.forward_grid_dim.value(),
            block_dim=self.forward_block_dim.value()
        )

        if self.training:
            self.computational_graph[].add_backward_operation_inputs(UnsafePointer(to=self), previous_layer, x_tensor.copy())

        return self.output_tensor.value(), self.output.value()

    fn backward(
        mut self,
        previous_layer: Optional[UnsafePointer[Layer[Self.dtype], MutAnyOrigin]],
        x_tensor_raw: LayerFuncTypeConstants[Self.dtype].LayerInputType,
        grad_output_tensor_raw: LayerFuncTypeConstants[Self.dtype].LayerGradOutputType
    ) raises -> Tuple[
        Tuple[LayoutTensor[Self.dtype, Self.X_GRAD_LAYOUT, MutAnyOrigin], DeviceBuffer[Self.dtype]],
        List[DeviceBuffer[Self.dtype]]
    ]:
        x_tensor = x_tensor_raw[LayoutTensor[Self.dtype, Self.X_LAYOUT, MutAnyOrigin]]
        grad_output_tensor = grad_output_tensor_raw[LayoutTensor[Self.dtype, Self.GRAD_OUTPUT_LAYOUT, MutAnyOrigin]]

        if not self.training:
            raise Error('Cannot backward if training is disabled')
        if not self.backward_memory_allocated:
            raise Error('Kernel memory must be allocated before calling the backward method')

        self.ctx.value().enqueue_function[Self.BACKWARD_KERNEL, Self.BACKWARD_KERNEL](
            self.x_gradient_tensor.value(),
            self.w_gradient_tensor.value(),
            self.b_gradient_tensor.value(),
            x_tensor,
            self.w_tensor.value(),
            grad_output_tensor,
            self.batch_size.value(),
            self.output_neurons,
            self.input_neurons,
            grid_dim=self.backward_grid_dim.value(),
            block_dim=self.backward_block_dim.value()
        )

        return (
            (self.x_gradient_tensor.value(), self.x_gradient.value()),
            [self.w_gradient.value(), self.b_gradient.value()]
        )
