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
    # The dataset is ordered, so it has to be shuffled and split into training and test datasets
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

model.save("photomath.model")
