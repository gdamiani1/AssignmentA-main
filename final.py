from crop import crop
import keras.models
import numpy as np
import tensorflow

import keras.preprocessing.image
from eq_parser import solve

from sklearn.preprocessing import LabelEncoder
from PIL import Image


def predict(img_path: str) -> str:
    """
    Outputs the label of the image using the model
    """
    img_array = keras.preprocessing.image.img_to_array(Image.open(img_path))
    img_array /= 255
    prediction = model.predict(img_array.reshape(1, 32, 32, 3))

    inverted = label_encoder.inverse_transform([np.argmax(prediction)])
    print("Prediction %s, confidence: %.2f" % (inverted[0], np.max(prediction)))
    return inverted[0]


model = keras.models.load_model("photomath.model")  # Model file path
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("pmclasses.npy")  # Lables

elem = crop('testing.png')  # Input image path
final_results = []

for i in range(0, elem):
    final_results.append(predict(f'{i}.png'))

solve(final_results)
