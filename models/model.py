import os
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

"""
Pretrained model accepts image resolution of 224x224, 
therefore image is resized to this resolution
"""
def preprocess_image(image_path, desired_size=224):
    img = Image.open(image_path)
    img = img.resize((desired_size,) * 2, resample=Image.LANCZOS)
    img = np.array(img)
    img = img / 255.0  # Rescale the image like in training
    return img

"""
Function to build the model for inference on the image
"""
def build_model():
    """
    Pretrained model use Densenet as the base model in specific DenseNet121 is used
    """
    base_model = DenseNet121(
        weights="imagenet",  
        include_top=False,
        input_shape=(224, 224, 3),
    )

    """
    On top of Densenet average pooling layer, dropout layer and 
    finally dense layer with 5 classes with softmax activation is used  
    """
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(5, activation="softmax")(x)  # Softmax for multi-class classification

    model = Model(inputs=base_model.input, outputs=predictions)

    """
    Model parameters used in training:
    optimizer : adam
    learning rate : 10e-5
    """
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.00005), metrics=["accuracy"]
    )

    return model

"""
Images are classified into 5 categories based on severity using the pretrained model
Return the maximum probability class for given image
"""
def classify_image(image_path):
    # Build model used for classification
    model = build_model()
    
    # Load weights from pretrained model
    # Replace with the correct path to your trained weights if needed
    model.load_weights("/models/pretrained/model.h5")

    # Create preprocessed image to be evaluated and predict its class
    x_val = np.empty((1, 224, 224, 3), dtype=np.float32)
    x_val[0, :, :, :] = preprocess_image(image_path)
    y_val_pred = model.predict(x_val)

    # Return the maximum probability class
    return np.argmax(np.squeeze(y_val_pred))

# # Example usage
# image_path = "path/to/image.jpg"
# predicted_class = classify_image(image_path)
# print(f"Predicted class: {predicted_class}")
