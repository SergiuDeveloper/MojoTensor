# mojo-tensor

A GPU-accelerated deep learning framework written in [Mojo](https://www.modular.com/mojo). Provides tensor operations, automatic differentiation, and neural network layers with explicit GPU kernel implementations.

## Features

- **Dense (fully connected) layers** with GPU-accelerated forward and backward passes
- **Automatic differentiation** via a computational graph
- **GPU kernels** with shared memory tiling for matrix multiplication
- **Activation functions**: ReLU (forward + backward)
- **Loss functions**: Mean Squared Error (forward + backward)
- **SGD optimizer** with SIMD-optimized weight updates
- **Layer serialization/deserialization** for model persistence

## Requirements

- [Pixi](https://prefix.dev/) package manager
- Linux x86-64
- Mojo `>=0.26.2.0.dev2026020616`
- NVIDIA or AMD GPU

## Installation

```bash
pixi install
```

## Usage

```mojo
from src import Dense, ComputationalGraph, SGD

# Set up optimizer and computational graph
var optimizer = SGD[DType.float32](learning_rate=0.001)
var graph = ComputationalGraph(UnsafePointer.address_of(optimizer))

# Define layers
var layer1 = Dense(UnsafePointer.address_of(graph), "dense1", 784, 256)
var layer2 = Dense(UnsafePointer.address_of(graph), "dense2", 256, 1)

layer1.set_training(True)
layer2.set_training(True)

# Allocate GPU memory for a given batch size
layer1.allocate_kernel_memory(device_context, batch_size)
layer2.allocate_kernel_memory(device_context, batch_size)

# Forward pass
var h1_tensor, h1_buffer = layer1.forward(None, input_tensor)
var out_tensor, out_buffer = layer2.forward(UnsafePointer.address_of(layer1), h1_tensor)

# Backward pass and weight update
graph.backward(loss_gradient)
graph.update_weights()
```

## Example: MNIST

A full training example is provided in `examples/mnist/mnist_fc_example.mojo`. It trains a two-layer network (784 → 256 → 1) on the MNIST dataset using MSE loss and SGD.

```bash
pixi run mojo examples/mnist/mnist_fc_example.mojo
```

Hyperparameters used in the example:

| Parameter     | Value |
| ------------- | ----- |
| Batch size    | 64    |
| Learning rate | 0.001 |
| Epochs        | 10    |
| Hidden units  | 256   |

The example saves trained weights to a pickle file after training.

## Architecture

| Component            | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `Dense`              | Fully connected layer: `output = input @ W^T + b`           |
| `ComputationalGraph` | Stores forward pass info and drives backpropagation         |
| `SGD`                | Stochastic gradient descent with configurable learning rate |
| `dense_kernel`       | Tiled matrix multiplication using GPU shared memory         |
| `relu`               | Element-wise ReLU and its gradient                          |
| `mse`                | MSE loss and its gradient                                   |
