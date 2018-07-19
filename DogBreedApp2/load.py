"""Module to load trained model"""

import numpy as np
from keras.models import model_from_json, Model
from keras.layers import Input
from keras.applications.xception import Xception
from scipy.misc import imread, imresize,imshow
import tensorflow as tf


def init():
    json_file = open('final_model.json','r')
    model_head_json = json_file.read()
    json_file.close()
    model_head = model_from_json(model_head_json)
    model_head.load_weights('final_model.h5')
    model_body = Xception(include_top=False, weights='imagenet')
    model_input = Input(shape=(299,299,3))
    x = model_body(model_input)
    predictions = model_head(x)
    full_model = Model(input=model_input, output=predictions)
    full_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return full_model, graph