"""
Main function of the Assignment A

- for the sake of proving the program works, I included a testing.png file that I made in MS Paint

Requires:   Model folder path
            Dataset lables path
            Input image path

Output:     Each symbol image prediction
            Equation from the input image
            Steps(in case there is a bracket)
            Final solution
"""


from crop import crop
import keras.models
import numpy as np

import keras.preprocessing.image
from eq_parser import solve

from sklearn.preprocessing import LabelEncoder
from PIL import Image


def predict(img_path: str) -> str:
    """
    Input:  Single symbol image path        (string)
    Output: Prediction                      (string)
    """
    img_array = keras.preprocessing.image.img_to_array(Image.open(img_path))
    img_array /= 255
    prediction = model.predict(img_array.reshape(1, 32, 32, 3))

    prediction = label_encoder.inverse_transform([np.argmax(prediction)])

    # The code below prints the prediction of the model and the confidence of the predictions

    # print("Prediction %s      Confidence: %.2f" % (prediction[0], np.max(prediction)))

    return prediction[0]


# if __name__ == '__main__':
model = keras.models.load_model("photomath.model")      # Model file path
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("pmclasses.npy")       # Dataset Lables

elem = crop('testing.png')                              # Input image path
final_results = []

for i in range(0, elem):
    final_results.append(predict(f"element_{i}.png"))

solve(final_results)
