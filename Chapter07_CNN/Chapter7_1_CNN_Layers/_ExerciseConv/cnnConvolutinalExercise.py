import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D


def conv2D(  # noqa: N802
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    image_rows, image_cols   = image.shape
    kernel_rows, kernel_cols = kernel.shape
    
    kernel_rows_half = kernel_rows // 2  # division with remainder
    kernel_cols_half = kernel_cols // 2  # division with remainder
    
    image_rows_padded = image_rows + 2 * kernel_rows_half
    image_cols_padded = image_cols + 2 * kernel_cols_half
    
    image_padded = np.zeros(shape=(image_rows_padded, image_cols_padded), dtype=np.float32)
    
    image_padded[kernel_rows_half:image_rows_padded - 1, kernel_cols_half:image_cols_padded - 1] = image
    
    image_result = np.zeros(shape=(image_rows, image_cols), dtype=np.float32)
    
    for image_colum in range(0, image_cols):
        for image_row in range(0, image_rows):
            sum = 0.0
            
            for kernel_row_index in range(0, kernel_rows):
                for kernel_column_index in range(0, kernel_cols):
                    value_image  = image_padded[image_colum + kernel_column_index, image_row + kernel_row_index]
                    value_kernel = kernel[kernel_column_index, kernel_row_index]
                    
                    sum += value_image * value_kernel
            
            image_result[image_colum, image_row] = sum
                    
    return image_result


def main() -> None:
    image = np.arange(16)
    image = image.reshape((4, 4)).astype(np.float32)
    kernel = np.ones(shape=(3, 3))

    conv_image = conv2D(image, kernel)

    print(f"Prvious shape: {image.shape} current shape: {conv_image.shape}")
    print(f"Conved Image:\n{conv_image.squeeze()}")

    layer = Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding="same")
    layer.build((4, 4, 1))
    W, b = layer.get_weights()  # noqa: N806
    layer.set_weights([np.ones_like(W), np.zeros_like(b)])
    conv_image_tf = layer(image.reshape((1, 4, 4, 1))).numpy()
    print(f"Conved Image TF:\n{conv_image_tf.squeeze()}")
    if not np.allclose(conv_image.flatten(), conv_image_tf.flatten()):
        raise AssertionError

    _, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(conv_image, cmap="gray")
    axs[2].imshow(conv_image_tf.squeeze(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
