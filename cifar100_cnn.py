# CIFAR-100 Image Classification Using CNN
# 
# This project builds a Convolutional Neural Network CNN using Keras.
# The model is trained and tested on the CIFAR-100 dataset.
# CIFAR-100 contains 100 different image classes.

from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def main():
    # 
    # Step 1: Load the CIFAR-100 dataset
    # 
    # x_train and x_test contain the image data.
    # y_train and y_test contain the image labels.
    # CIFAR-100 has 50,000 training images and 10,000 test images.

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    print("Training data shape before normalization:", x_train.shape)
    print("Test data shape before normalization:", x_test.shape)

    # 
    # Step 2: Normalize the images
    # 
    # The pixel values are originally between 0 and 255.
    # Dividing by 255 changes the values to the range 0 to 1.
    # This helps the neural network train better.

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # 
    # Step 3: Encode the target labels
    # 
    # CIFAR-100 has 100 classes.
    # The labels are converted into categorical format using one-hot encoding.

    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    # 
    # Step 4: Build the CNN model
    # 
    # The model follows the required architecture:
    #
    # 1. Conv2D layer with 32 filters, 3x3 kernel, ReLU activation
    # 2. Conv2D layer with 32 filters, 3x3 kernel, ReLU activation
    # 3. MaxPooling2D layer with pool size 2x2
    # 4. Dropout layer with rate 0.25
    # 5. Two Conv2D layers, each with 64 filters, 3x3 kernel, ReLU activation
    # 6. MaxPooling2D layer with pool size 2x2
    # 7. Dropout layer with rate 0.25
    # 8. Flattening layer
    # 9. Dense layer with 512 units and ReLU activation
    # 10. Dropout layer with rate 0.5
    # 11. Output Dense layer with 100 units and Softmax activation

    model = Sequential()

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(32, 32, 3)
        )
    )

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        Dropout(0.25)
    )

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        Dropout(0.25)
    )

    model.add(Flatten())

    model.add(
        Dense(
            units=512,
            activation='relu'
        )
    )

    model.add(
        Dropout(0.5)
    )

    model.add(
        Dense(
            units=100,
            activation='softmax'
        )
    )

    # 
    # Step 5: Compile the model
    #
    # Optimizer: Adam
    # Loss function: categorical_crossentropy
    # Metric: accuracy

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print the model architecture
    model.summary()

    # 
    # Step 6: Train the model
    # 
    # The model is trained using the training images and labels.
    # Required epochs: 100
    # Required batch size: 32

    model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=32
    )

    # 
    # Step 7: Test the model
    # 
    # The model is evaluated using the test set.
    # The result contains test loss and test accuracy.

    test_results = model.evaluate(x_test, y_test)

    print("Test loss:", test_results[0])
    print("Test accuracy:", test_results[1])

    # The project requirement says to return the model and test results.
    return model, test_results


if __name__ == "__main__":
    main()
