import os
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE
from python import Python
from math import sqrt
from memory import memcpy

from src.layers import Dense, Layer, LayerFuncTypeConstants, save_layers
from src.optimizers import Optimizer, SGD
from src.computational_graph import ComputationalGraph
from src.kernels.losses import mse_forward, mse_backward
from src.kernels.constants import MAX_GRID_SIZE

fn download_if_missing(url: String, dest: String) raises:
    if os.path.exists(dest):
        return
    print(1)
    urllib_m = Python.import_module('urllib.request')
    print('Downloading', url)
    urllib_m.urlretrieve(url, dest)

fn extract_if_missing(zip_path: String, csv_path: String, extract_dir: String) raises:
    if os.path.exists(csv_path):
        return
    print(2)
    zipfile_m = Python.import_module('zipfile')
    z = zipfile_m.ZipFile(zip_path, 'r')
    z.extractall(extract_dir)
    z.close()

# Returns (pixels, labels) where pixels is a flat List[Float32] of shape
# (n_samples * 784,) normalised to [0, 1], and labels is List[Int].
# Parses byte-by-byte to avoid creating millions of temporary String objects.
fn load_mnist(csv_path: String) raises -> Tuple[List[Float32], List[Int]]:
    pixels = List[Float32](capacity=60000 * 784)
    labels = List[Int](capacity=60000)

    with open(csv_path, 'r') as f:
        content = f.read()

    ptr = content.unsafe_ptr()
    n = len(content)
    i = 0
    first_line = True

    while i < n:
        # Skip '\n'/'\r' between rows
        while i < n and (ptr[i] == 10 or ptr[i] == 13):
            i += 1
        if i >= n:
            break

        # Skip header row if present (starts with 'l' for "label")
        if first_line:
            first_line = False
            if ptr[i] == 108:  # ord('l') == 108
                while i < n and ptr[i] != 10:
                    i += 1
                continue

        # Parse label (first CSV field)
        label_val: Int = 0
        while i < n and ptr[i] != 44 and ptr[i] != 10 and ptr[i] != 13:
            label_val = label_val * 10 + Int(ptr[i]) - 48
            i += 1
        labels.append(label_val)
        if i < n and ptr[i] == 44:  # skip ','
            i += 1

        # Parse 784 pixel values
        for j in range(784):
            pixel_val: Int = 0
            while i < n and ptr[i] != 44 and ptr[i] != 10 and ptr[i] != 13:
                pixel_val = pixel_val * 10 + Int(ptr[i]) - 48
                i += 1
            pixels.append(Float32(pixel_val) / 255.0)
            if i < n and ptr[i] == 44:  # skip ','
                i += 1

        # Skip any remaining bytes on this row
        while i < n and ptr[i] != 10:
            i += 1

    return pixels^, labels^

fn main() raises:
    comptime DTYPE = DType.float32
    comptime INPUT_NEURONS = 784
    comptime HIDDEN_NEURONS = 256
    comptime OUTPUT_NEURONS = 1      # regression: predict label/9 in [0, 1]
    comptime BATCH_SIZE = 64
    comptime EPOCHS = 10
    comptime LEARNING_RATE = 0.001

    comptime X_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime GRAD_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime MSE_LAYOUT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)

    # MSE kernel: treat single output as (BATCH_SIZE, 1, 1)
    comptime MSE_TPB = 16
    comptime MSE_BLOCKS_PER_BATCH = ((OUTPUT_NEURONS + MSE_TPB - 1) // MSE_TPB) * ((1 + MSE_TPB - 1) // MSE_TPB)
    comptime MSE_TOTAL_BLOCKS = BATCH_SIZE * MSE_BLOCKS_PER_BATCH
    comptime MSE_GRID_X = min(MSE_TOTAL_BLOCKS, MAX_GRID_SIZE)
    comptime MSE_GRID_Y = (MSE_TOTAL_BLOCKS + MSE_GRID_X - 1) // MSE_GRID_X
    comptime MSE_GRID_DIM = (MSE_GRID_X, MSE_GRID_Y, 1)
    comptime MSE_BLOCK_DIM = (MSE_TPB, MSE_TPB)
    comptime MSE_FWD_KERNEL = mse_forward[MSE_TPB, DTYPE, MSE_LAYOUT, MSE_LAYOUT, MSE_LAYOUT]
    comptime MSE_BWD_KERNEL = mse_backward[MSE_TPB, DTYPE, MSE_LAYOUT, MSE_LAYOUT, MSE_LAYOUT, MSE_LAYOUT]

    MNIST_DIR = 'examples/mnist'
    TRAIN_URL = 'https://github.com/phoebetronic/mnist/raw/main/mnist_train.csv.zip'
    TEST_URL  = 'https://github.com/phoebetronic/mnist/raw/main/mnist_test.csv.zip'
    TRAIN_ZIP = MNIST_DIR + '/mnist_train.csv.zip'
    TEST_ZIP  = MNIST_DIR + '/mnist_test.csv.zip'
    TRAIN_CSV = MNIST_DIR + '/mnist_train.csv'
    TEST_CSV  = MNIST_DIR + '/mnist_test.csv'
    WEIGHTS_PATH = MNIST_DIR + '/weights.pkl'

    download_if_missing(TRAIN_URL, TRAIN_ZIP)
    download_if_missing(TEST_URL,  TEST_ZIP)
    extract_if_missing(TRAIN_ZIP, TRAIN_CSV, MNIST_DIR)
    extract_if_missing(TEST_ZIP,  TEST_CSV,  MNIST_DIR)

    print('Loading training data...')
    var train_loaded = load_mnist(TRAIN_CSV)
    train_pixels = train_loaded[0].copy()
    train_labels = train_loaded[1].copy()
    _ = train_loaded^
    print('Loading test data...')
    var test_loaded = load_mnist(TEST_CSV)
    test_pixels = test_loaded[0].copy()
    test_labels = test_loaded[1].copy()
    _ = test_loaded^

    n_train = len(train_labels)
    n_test  = len(test_labels)
    print('Train:', n_train, '  Test:', n_test)

    optimizer = SGD[DTYPE](LEARNING_RATE)
    optimizer_ptr = UnsafePointer(to=optimizer)
    computational_graph = ComputationalGraph[DTYPE](
        UnsafePointer[Optimizer[DTYPE], MutAnyOrigin](optimizer_ptr)
    )
    computational_graph_ptr = UnsafePointer(to=computational_graph)

    layer1 = Dense[DTYPE](computational_graph_ptr, 'dense1', INPUT_NEURONS, HIDDEN_NEURONS)
    layer2 = Dense[DTYPE](computational_graph_ptr, 'dense2', HIDDEN_NEURONS, OUTPUT_NEURONS)

    # Xavier uniform init: weights in [-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]
    # rand() gives [0,1), so transform: w = (w*2 - 1) * scale
    with layer1.w_cpu.map_to_host() as w_host:
        scale1 = sqrt(6.0 / Float64(INPUT_NEURONS + HIDDEN_NEURONS)).cast[DTYPE]()
        for i in range(INPUT_NEURONS * HIDDEN_NEURONS):
            w_host[i] = (w_host[i] * 2 - 1) * scale1
    with layer1.b_cpu.map_to_host() as b_host:
        for i in range(HIDDEN_NEURONS):
            b_host[i] = Scalar[DTYPE](0)

    with layer2.w_cpu.map_to_host() as w_host:
        scale2 = sqrt(6.0 / Float64(HIDDEN_NEURONS + OUTPUT_NEURONS)).cast[DTYPE]()
        for i in range(HIDDEN_NEURONS * OUTPUT_NEURONS):
            w_host[i] = (w_host[i] * 2 - 1) * scale2
    with layer2.b_cpu.map_to_host() as b_host:
        for i in range(OUTPUT_NEURONS):
            b_host[i] = Scalar[DTYPE](0)

    layer1.set_training(True)
    layer2.set_training(True)

    with DeviceContext() as ctx:
        layer1.allocate_kernel_memory(ctx, BATCH_SIZE)
        layer2.allocate_kernel_memory(ctx, BATCH_SIZE)

        # Input buffer reused every batch
        x_buf = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * INPUT_NEURONS)
        x_tensor = LayoutTensor[DTYPE, X_LAYOUT, MutAnyOrigin](
            x_buf,
            RuntimeLayout[X_LAYOUT](
                RuntimeTuple[X_LAYOUT.shape](BATCH_SIZE, INPUT_NEURONS),
                RuntimeTuple[X_LAYOUT.stride](INPUT_NEURONS, 1)
            )
        )

        # MSE buffers: one scalar per sample, so BATCH_SIZE elements total
        targets_buf = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_NEURONS)
        mse_out_buf = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_NEURONS)
        mse_grad_buf = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_NEURONS)
        local_grad_buf = ctx.enqueue_create_buffer[DTYPE](BATCH_SIZE * OUTPUT_NEURONS)

        # local_gradient = 1/BATCH_SIZE for mean MSE (one output per sample)
        local_grad_buf.enqueue_fill((1.0 / Float64(BATCH_SIZE)).cast[DTYPE]())

        # 3D views: (BATCH_SIZE, 1, OUTPUT_NEURONS) over each buffer
        targets_tensor = LayoutTensor[DTYPE, MSE_LAYOUT, ImmutAnyOrigin](
            targets_buf,
            RuntimeLayout[MSE_LAYOUT](
                RuntimeTuple[MSE_LAYOUT.shape](BATCH_SIZE, 1, OUTPUT_NEURONS),
                RuntimeTuple[MSE_LAYOUT.stride](OUTPUT_NEURONS, OUTPUT_NEURONS, 1)
            )
        )
        mse_out_tensor = LayoutTensor[DTYPE, MSE_LAYOUT, MutAnyOrigin](
            mse_out_buf,
            RuntimeLayout[MSE_LAYOUT](
                RuntimeTuple[MSE_LAYOUT.shape](BATCH_SIZE, 1, OUTPUT_NEURONS),
                RuntimeTuple[MSE_LAYOUT.stride](OUTPUT_NEURONS, OUTPUT_NEURONS, 1)
            )
        )
        mse_grad_tensor = LayoutTensor[DTYPE, MSE_LAYOUT, MutAnyOrigin](
            mse_grad_buf,
            RuntimeLayout[MSE_LAYOUT](
                RuntimeTuple[MSE_LAYOUT.shape](BATCH_SIZE, 1, OUTPUT_NEURONS),
                RuntimeTuple[MSE_LAYOUT.stride](OUTPUT_NEURONS, OUTPUT_NEURONS, 1)
            )
        )
        local_grad_tensor = LayoutTensor[DTYPE, MSE_LAYOUT, ImmutAnyOrigin](
            local_grad_buf,
            RuntimeLayout[MSE_LAYOUT](
                RuntimeTuple[MSE_LAYOUT.shape](BATCH_SIZE, 1, OUTPUT_NEURONS),
                RuntimeTuple[MSE_LAYOUT.stride](OUTPUT_NEURONS, OUTPUT_NEURONS, 1)
            )
        )

        # 3D view over layer2's output for MSE (same memory, reinterpreted as 3D)
        pred_tensor_3d = LayoutTensor[DTYPE, MSE_LAYOUT, ImmutAnyOrigin](
            layer2.output.value(),
            RuntimeLayout[MSE_LAYOUT](
                RuntimeTuple[MSE_LAYOUT.shape](BATCH_SIZE, 1, OUTPUT_NEURONS),
                RuntimeTuple[MSE_LAYOUT.stride](OUTPUT_NEURONS, OUTPUT_NEURONS, 1)
            )
        )

        # 2D view over mse_grad_buf to feed into computational_graph.backward()
        mse_grad_tensor_2d = LayoutTensor[DTYPE, GRAD_LAYOUT, MutAnyOrigin](
            mse_grad_buf,
            RuntimeLayout[GRAD_LAYOUT](
                RuntimeTuple[GRAD_LAYOUT.shape](BATCH_SIZE, OUTPUT_NEURONS),
                RuntimeTuple[GRAD_LAYOUT.stride](OUTPUT_NEURONS, 1)
            )
        )

        for epoch in range(EPOCHS):
            # --- Training ---
            layer1.set_training(True)
            layer2.set_training(True)

            n_train_batches = n_train // BATCH_SIZE
            train_loss_sum: Float64 = 0.0
            train_correct: Int = 0

            for batch_idx in range(n_train_batches):
                batch_start = batch_idx * BATCH_SIZE

                with x_buf.map_to_host() as x_host, targets_buf.map_to_host() as targets_host:
                    pixels_ptr = train_pixels.unsafe_ptr()
                    x_ptr = x_host.unsafe_ptr()
                    targets_ptr = targets_host.unsafe_ptr()
                    for b in range(BATCH_SIZE):
                        memcpy(dest=x_ptr + b * INPUT_NEURONS, src=pixels_ptr + (batch_start + b) * INPUT_NEURONS, count=INPUT_NEURONS)
                        targets_ptr[b] = Float32(train_labels[batch_start + b]) / 9.0

                h1_tensor, _ = layer1.forward(None, x_tensor)
                _, out_buf = layer2.forward(
                    UnsafePointer[Dense[DTYPE], MutAnyOrigin](UnsafePointer(to=layer1)),
                    h1_tensor
                )

                # MSE forward + backward entirely on GPU
                ctx.enqueue_function[MSE_FWD_KERNEL, MSE_FWD_KERNEL](
                    mse_out_tensor, pred_tensor_3d, targets_tensor,
                    BATCH_SIZE, 1, OUTPUT_NEURONS,
                    grid_dim=MSE_GRID_DIM, block_dim=MSE_BLOCK_DIM
                )
                ctx.enqueue_function[MSE_BWD_KERNEL, MSE_BWD_KERNEL](
                    mse_grad_tensor, pred_tensor_3d, targets_tensor, local_grad_tensor,
                    BATCH_SIZE, 1, OUTPUT_NEURONS,
                    grid_dim=MSE_GRID_DIM, block_dim=MSE_BLOCK_DIM
                )
                ctx.synchronize()

                # CPU: sum the per-sample squared errors for the scalar loss,
                # and argmax (round pred*9) for accuracy
                batch_loss: Float64 = 0.0
                with mse_out_buf.map_to_host() as mse_out_host, out_buf.map_to_host() as out_host:
                    mse_ptr = mse_out_host.unsafe_ptr()
                    out_ptr = out_host.unsafe_ptr()
                    labels_ptr = train_labels.unsafe_ptr()
                    for b in range(BATCH_SIZE):
                        batch_loss += Float64(mse_ptr[b])
                        pred_label = max(0, min(9, Int(Float64(out_ptr[b]) * 9.0 + 0.5)))
                        if pred_label == labels_ptr[batch_start + b]:
                            train_correct += 1
                train_loss_sum += batch_loss / Float64(BATCH_SIZE)

                grad_variant = LayerFuncTypeConstants[DTYPE].LayerGradOutputType(mse_grad_tensor_2d)
                computational_graph.backward(grad_variant)
                ctx.synchronize()
                computational_graph.update_weights()
            train_loss = train_loss_sum / Float64(n_train_batches)
            train_acc  = Float64(train_correct) / Float64(n_train_batches * BATCH_SIZE)

            # --- Evaluation ---
            layer1.set_training(False)
            layer2.set_training(False)

            n_test_batches = n_test // BATCH_SIZE
            test_loss_sum: Float64 = 0.0
            test_correct: Int = 0

            for batch_idx in range(n_test_batches):
                batch_start = batch_idx * BATCH_SIZE

                with x_buf.map_to_host() as x_host, targets_buf.map_to_host() as targets_host:
                    test_pixels_ptr = test_pixels.unsafe_ptr()
                    test_x_ptr = x_host.unsafe_ptr()
                    test_targets_ptr = targets_host.unsafe_ptr()
                    for b in range(BATCH_SIZE):
                        memcpy(dest=test_x_ptr + b * INPUT_NEURONS, src=test_pixels_ptr + (batch_start + b) * INPUT_NEURONS, count=INPUT_NEURONS)
                        test_targets_ptr[b] = Float32(test_labels[batch_start + b]) / 9.0

                h1_tensor, _ = layer1.forward(None, x_tensor)
                _, out_buf = layer2.forward(
                    UnsafePointer[Dense[DTYPE], MutAnyOrigin](UnsafePointer(to=layer1)),
                    h1_tensor
                )

                ctx.enqueue_function[MSE_FWD_KERNEL, MSE_FWD_KERNEL](
                    mse_out_tensor, pred_tensor_3d, targets_tensor,
                    BATCH_SIZE, 1, OUTPUT_NEURONS,
                    grid_dim=MSE_GRID_DIM, block_dim=MSE_BLOCK_DIM
                )
                ctx.synchronize()

                with mse_out_buf.map_to_host() as mse_out_host, out_buf.map_to_host() as out_host:
                    test_mse_ptr = mse_out_host.unsafe_ptr()
                    test_out_ptr = out_host.unsafe_ptr()
                    test_labels_ptr = test_labels.unsafe_ptr()
                    for b in range(BATCH_SIZE):
                        test_loss_sum += Float64(test_mse_ptr[b])
                        pred_label = max(0, min(9, Int(Float64(test_out_ptr[b]) * 9.0 + 0.5)))
                        if pred_label == test_labels_ptr[batch_start + b]:
                            test_correct += 1

            test_loss = test_loss_sum / Float64(n_test_batches * BATCH_SIZE)
            test_acc  = Float64(test_correct) / Float64(n_test_batches * BATCH_SIZE)

            print(
                'Epoch', epoch + 1, '/', EPOCHS,
                ' train_loss =', train_loss,
                ' train_acc =', train_acc,
                ' test_loss =', test_loss,
                ' test_acc =', test_acc
            )

        # --- Save weights ---
        layers_to_save = List[UnsafePointer[Layer[DTYPE], MutAnyOrigin]]()
        layers_to_save.append(UnsafePointer(to=layer1))
        layers_to_save.append(UnsafePointer(to=layer2))
        save_layers[DTYPE](layers_to_save, WEIGHTS_PATH)
        print('Weights saved to', WEIGHTS_PATH)
