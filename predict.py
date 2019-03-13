from keras.models import load_model
import cv2
import numpy as np

model = load_model('./saved_models/faces95/faces95ModelMBDSelu.h5')

model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])

img = cv2.imread('test.png')
img = np.reshape(img,[1,1, 71, 64])

classes = model.predict_classes(img)

print(classes)