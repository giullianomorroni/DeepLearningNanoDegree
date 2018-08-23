import cv2
from keras.models import load_model
import numpy as np

model = load_model('model.weights.best.hdf5')

file_path = './dogs/001.Affenpinscher/Affenpinscher_00001.jpg'
im = cv2.imread(file_path)
print(im.shape)
im = np.divide(im, 255)

#file_path = './human-faces/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
#im = cv2.imread(file_path)
#print(im.shape)
#im = np.divide(im, 255)

y_hat = model.predict([[im]])
print(y_hat)
