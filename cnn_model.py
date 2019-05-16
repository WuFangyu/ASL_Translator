from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D 
from tensorflow.keras.models import Sequential
import tensorflow as tf
import keras
import h5py

def create_model():
    my_model = Sequential()
    my_model.add(Conv2D(32, kernel_size=2, strides=1, input_shape=target_dims, padding="SAME"))
    my_model.add(LeakyReLU())
    my_model.add(MaxPooling2D(pool_size=(2,2)))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(128, kernel_size=3, strides=1, padding="SAME"))
    my_model.add(LeakyReLU())
    my_model.add(MaxPooling2D(pool_size=(3,3)))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(128, kernel_size=4, strides=1, padding="SAME"))
    my_model.add(LeakyReLU())
    my_model.add(MaxPooling2D(pool_size=(4,4)))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(256, kernel_size=4, strides=1, padding="SAME"))
    my_model.add(LeakyReLU())
    my_model.add(MaxPooling2D(pool_size=(2,2)))
    my_model.add(Flatten())
    my_model.add(Dropout(0.5))
    my_model.add(Dense(512, activation='relu'))
    my_model.add(Dense(num_classes, activation='softmax'))
    my_model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=["accuracy"])
    return my_model

def fetch_model():
	f = h5py.File('cur_model.h5py', 'r')
	model = tf.keras.models.load_model(f)
	return model

