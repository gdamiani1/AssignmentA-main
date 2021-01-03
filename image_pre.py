"""
The file contains the code that processes the dataset and traines the model
    - not required to run the main function (final.py)

Requires:       Dataset CSV file path

Outputs:        Model file
                Dataset classes file
"""

import keras.preprocessing.image
import keras.callbacks
import csv
import random
import numpy as np

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def dataset_prep(path: str):
    """
    Prepares the dataset
        Reads the CSV file in order to find the image file names and sort labels

    Dataset used: HASYv2


    Input: dataset path     (string)
    Output: saves the dataset labels in a 'pmclasses.npy' file
    """
    images = []
    classes = []

    with open(path) as csvfile:
        csvreader = csv.reader(csvfile)
        i = 0
        for row in csvreader:
            if i > 0:
                image = keras.preprocessing.image.img_to_array(Image.open('hasyv2/' + row[0]))
                image /= 255.0
                images.append((row[0], row[2], image))
                classes.append(row[2])
            i += 1

    random.shuffle(images)

    train = images

    traininput = np.asarray(list(map(lambda tirow: row[2], train)))

    trainoutput = np.asarray(list(map(lambda torow: row[1], train)))

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(classes)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)

    trainoutput_int = label_encoder.transform(trainoutput)
    trainoutput = onehot_encoder.transform(trainoutput_int.reshape(len(trainoutput_int), 1))

    numclasses = len(label_encoder.classes_)

    np.save('pmclasses.npy', label_encoder.classes_)

    return traininput, trainoutput, numclasses


def model_build():
    """
    Builds a model and saves it as a photomath.model file

    Output:
        Epoch 1/10
        4206/4206 - 110s - loss: 1.4008 - accuracy: 0.6544 - val_loss: 0.9312 - val_accuracy: 0.7465
        Epoch 2/10
        4206/4206 - 118s - loss: 0.8990 - accuracy: 0.7477 - val_loss: 0.8582 - val_accuracy: 0.7589
        Epoch 3/10
        4206/4206 - 110s - loss: 0.7937 - accuracy: 0.7694 - val_loss: 0.8175 - val_accuracy: 0.7697
        Epoch 4/10
        4206/4206 - 111s - loss: 0.7272 - accuracy: 0.7845 - val_loss: 0.8205 - val_accuracy: 0.7651
        Epoch 5/10
        4206/4206 - 112s - loss: 0.6814 - accuracy: 0.7916 - val_loss: 0.8034 - val_accuracy: 0.7736
        Epoch 6/10
        4206/4206 - 107s - loss: 0.6390 - accuracy: 0.8018 - val_loss: 0.8155 - val_accuracy: 0.7629
        Epoch 7/10
        4206/4206 - 105s - loss: 0.6083 - accuracy: 0.8079 - val_loss: 0.8410 - val_accuracy: 0.7660
        Epoch 8/10
        4206/4206 - 106s - loss: 0.5819 - accuracy: 0.8150 - val_loss: 0.8429 - val_accuracy: 0.7682
        Epoch 9/10
        4206/4206 - 109s - loss: 0.5617 - accuracy: 0.8194 - val_loss: 0.8529 - val_accuracy: 0.7690
        Epoch 10/10
        4206/4206 - 111s - loss: 0.5425 - accuracy: 0.8231 - val_loss: 0.8967 - val_accuracy: 0.7614
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import MaxPooling2D, Conv2D

    pmmodel = Sequential()
    pmmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    pmmodel.add(MaxPooling2D(pool_size=(2, 2)))
    pmmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    pmmodel.add(MaxPooling2D(pool_size=(2, 2)))
    pmmodel.add(Flatten())
    pmmodel.add(Dense(1024, activation='tanh'))
    pmmodel.add(Dropout(0.5))
    pmmodel.add(Dense(num_classes, activation='softmax'))

    pmmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    pmmodel.fit(train_input, train_output,
                batch_size=32,
                epochs=10,
                verbose=2,
                validation_split=0.2)

    return pmmodel


dataset_path = 'hasyv2/hasy-data-labels.csv'

train_input, train_output, num_classes = dataset_prep(dataset_path)
print('data loaded')

model = model_build()

model.save("pm.model")
