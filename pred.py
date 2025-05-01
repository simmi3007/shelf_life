import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

IMG_SIZE = 224

# Load the saved model
model_path = 'Models/full_model.h5'
model = tf.keras.models.load_model(model_path)

# Define the prediction function
def pred(img_path):
    
    # Load and preprocess the image
    def load_and_preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale to [0, 1]
        return img_array

    # Load the image and preprocess it
    img_array = load_and_preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Use the class indices from the train_generator to map class indices to labels
    #class_labels = list(train_generator.class_indices.keys())  
    class_labels=['0', '1', '1-2', '1-3', '2-5', '3-6', '4-8', '6-9', '7-10', '7-12', '9-12', '9-14']
    
    predicted_label = class_labels[predicted_class[0]]

    # Display the input image along with the predicted class label
    display_image_with_prediction(img_path, predicted_label)
    return predicted_label

# Function to display image along with the predicted label
def display_image_with_prediction(img_path, predicted_label):
    # Load and display the image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    plt.imshow(img)
    plt.title(f'Predicted shelf life in days: {predicted_label}')
    plt.axis('off')  # Turn off the axis
    plt.show()

# Test the function with a sample image path
'''img_path = 'C:\\Users\\HP\\Downloads\\Fruits Detection and Quality Analysis.v1i.tensorflow\\Banana.jpeg'  # Replace with your image path 
print(f'Predicted class: {pred(img_path)}')'''
