import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import MaxPool2D


def max_pooling(
    image: np.ndarray,
) -> np.ndarray:
    rows_image, cols_image = image.shape
    
    cols_output = 14
    rows_output = 14
    
    cols_kernel = 2
    rows_kernel = 2
    
    stride = 2
    
    output_array = np.zeros(shape=(rows_output, cols_output), dtype=np.float32)
    
    row_counter = 0
    
    for row_index in range(0, rows_image, stride):
        col_counter = 0
        
        for col_index in range(0, cols_image, stride):
            patch     = image[row_index: row_index + stride, col_index:col_index + stride]
            max_value = np.max(patch)
            
            output_array[row_counter, col_counter] = max_value
            
            col_counter += 1
            
        row_counter += 1
    
    return output_array


def main() -> None:
    (x_train, _), (_, _) = mnist.load_data()

    image = x_train[0]
    image = image.reshape((28, 28)).astype(np.float32)

    pooling_image = max_pooling(image)

    print(f"Prvious shape: {image.shape} current shape: {pooling_image.shape}")
    print(f"Pooled Image:\n{pooling_image.squeeze()}")

    layer = MaxPool2D(pool_size=(2, 2), strides=2, padding="valid")
    pooling_image_tf = layer(image.reshape((1, 28, 28, 1))).numpy()
    print(f"Pooled Image TF:\n{pooling_image_tf.squeeze()}")
    if not np.allclose(pooling_image.flatten(), pooling_image_tf.flatten()):
        raise AssertionError

    _, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(pooling_image, cmap="gray")
    axs[2].imshow(pooling_image_tf.squeeze(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
