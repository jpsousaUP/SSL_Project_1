import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Model


n_channels = 2 
resolution = 320
n_classes = 4

def build_simpleCNN_model(resolution, n_channels):
    """
        Build a simple CNN model with 2 convolutional layers, 2 max pooling layers, 1 fully connected layer and 1 output layer.
        The model is built using the Keras functional API.
        The model is compiled with the Adam optimizer and the categorical cross entropy loss function.
    """
    
    # Define the input shape
    input_shape = Input(shape=(resolution, resolution, n_channels))
    x = Conv2D(4, (3, 3), activation='relu')(input_shape) # 32
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu')(x) # 64
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Add a fully connected layer with 128 units and a ReLU activation function
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.5)(x)
    # Add two output layers each with 1 unit and a linear activation function
    output1 = Dense(n_classes, activation="softmax")(x)

    # Define the model with the specified input and outputs
    model = Model(inputs=input_shape, outputs=output1)
    return model


# Test the function    
""" model = build_simpleCNN_model(resolution, n_channels)

# Print the model summary
model.summary()
 """