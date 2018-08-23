import cv2
import os
from pathlib import Path
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

main_path = './human-faces/'

X_test, y_test, X_valid, y_valid, X_train, y_train = [],[],[],[],[],[]


def load_data():
    data = []
    for directory in os.listdir(main_path):
        inside = os.listdir(main_path + directory)
        for file in inside:
            file_path = main_path + directory + '/' + file
            im = cv2.imread(file_path)

            im = np.divide(im, 255)
            data.append(im)

    print(data[0].shape)

    X_test = data[0:30]
    y_test = [[1,0] for x in range(0, 30)]

    X_valid = data[30:60]
    y_valid = [[1,0] for x in range(30, 60)]

    X_train = data[60:200]
    y_train = [[1,0] for x in range(0, len(data[60:200]))]

    np.save('X_train', X_train, True)
    np.save('y_train', y_train, True)

    np.save('X_valid', X_valid, True)
    np.save('y_valid', y_valid, True)

    np.save('X_test', X_test, True)
    np.save('y_test', y_test, True)


if Path('X_train.npy').is_file():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')

    X_valid = np.load('X_valid.npy')
    y_valid = np.load('y_valid.npy')

    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
else:
    load_data()

model = Sequential()

model.add(Conv2D(filters=2, kernel_size=2, padding='valid', activation='relu', input_shape=(250, 250, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=4, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

#model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=2))

#model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(2, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train the model
check_pointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)


hist = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_valid, y_valid),
                 callbacks=[check_pointer], verbose=2, shuffle=True)

model.load_weights('model.weights.best.hdf5')
score = model.evaluate(X_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])
