from cnn_model import fetch_model
import numpy as np
import pandas as pd 
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from glob import glob
import random
from keras.preprocessing import image
import tensorflow as tf

labels = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

model1 = fetch_model()
model1.summary()

# check if cap is open
cap = cv2.VideoCapture(0)
print(cap.isOpened())

timeInterval = 0

# openCV window
while True:
    ret, frame = cap.read()

    # convert image to grey scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 =  10, 200, 310, 500
    img = frame
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    test_img = img[y1:y2, x1:x2]
    cv2.imwrite("cur_img.jpg", test_img)

    if timeInterval == 30:
        test_img = image.load_img("cur_img.jpg", target_size=(64, 64))
        x = image.img_to_array(test_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        class_res = model1.predict_classes(x)
        print(labels[class_res[0]])
        timeInterval = 0

    if cv2.waitKey(1) == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()
