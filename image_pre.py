import keras.preprocessing.image
import keras.callbacks
import csv
import random
import numpy as np
import cv2 as cv
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder




def dataset_prep(path: str):
    """
    Takes in a dataset path and creates two arrays, train_input, containing images converted into a normalized array,
    and test_output, containing image labels.
    Outputs the two arrays and the number of classes of the dataset
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
    #The dataset is ordered, so it has to be shuffled and split into training and test datasets
    random.shuffle(images)
    #split = int(len(images) * 0.8)
    train = images#[:split]
    #test = images[split:]

    train_input = np.asarray(list(map(lambda row: row[2], train)))

    train_output = np.asarray(list(map(lambda row: row[1], train)))


    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(classes)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)

    train_output_int = label_encoder.transform(train_output)
    train_output = onehot_encoder.transform(train_output_int.reshape(len(train_output_int), 1))

    num_classes = len(label_encoder.classes_)

    np.save('pmclasses.npy', label_encoder.classes_)

    return train_input, train_output, num_classes


def model_build():
    """
    Returns the model
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import MaxPooling2D, Conv2D

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/mnist-style')

    model.fit(train_input, train_output,
              batch_size=32,
              epochs=10,
              verbose=2,
              validation_split=0.2,
              callbacks=[tensorboard])

    return model

if __name__ == '__main__':
    dataset_path = 'hasyv2/hasy-data-labels.csv'

    train_input, train_output, num_classes = dataset_prep(dataset_path)
    print('data loaded')

    model = model_build()

    model.save("photomath.model")




