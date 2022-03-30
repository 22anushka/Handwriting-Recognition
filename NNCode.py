import sys
import tensorflow as tf

# Using MNIST Handwriting dataset
mnist = tf.keras.datasets.mnist

# Prepare data for training :- X_train is all the instance with attributes, y_train is the label of each instance
# Load the data, normalize the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# convert to binary matrix : required to train model i.e spam or not spam
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.util.to_categorical(y_test)

# reshapes:  Flatten images to 1-D vector
# shape gives indication of number of dimensions
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

# Create a CNN
model = tf.keras.model.Sequential(
    [
        # Convolution layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(28, 28, 1)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Droupout(0.5),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(10, activation="softmax")
    
    ]
)

# Train the neural network
model.compile(
    optimizer = "adam",
    loss = "categorial_crossentropy", # for multi-cass classification tasks
    metrics = ["accuracy"]
)
# to measure how well data is generalized, epoch = number of passes to complete
model.fit(x_train, y_train, epochs=10)

# Evaluate NN performance
model.evaluate(x_test, y_test, verbose=2)

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")