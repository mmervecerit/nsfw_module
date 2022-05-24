#! python

import json
import os
from os import listdir
from os.path import isfile, join, exists, isdir, abspath

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


IMAGE_DIM = 224   # required/default image dimensionality



def load_images(image_paths, image_size, verbose=True):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
        verbose: show all of the image path and sizes loaded
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    '''
    loaded_images = []
    loaded_image_paths = []

    if isdir(image_paths):
        parent = abspath(image_paths)
        image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
    elif isfile(image_paths):
        image_paths = [image_paths]

    for img_path in image_paths:
        try:
            if verbose:
                print(img_path, "size:", image_size)
            image = keras.preprocessing.image.load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print("Image Load Failure: ", img_path, ex)
    
    return np.asarray(loaded_images), loaded_image_paths


def load_model():
    tf.config.threading.set_inter_op_parallelism_threads(15)
    tf.config.threading.set_intra_op_parallelism_threads(15)
    tf.config.set_soft_device_placement(True)
    model_path = './nsfw_model.h5'
    if not os.access(model_path, os.W_OK):
        os.system('wget http://github.com/mmervecerit/nsfw_module/raw/main/nsfw_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    return model


def classify(model, input_paths, image_dim=IMAGE_DIM):
    """ Classify given a model, input paths (could be single string), and image dimensionality...."""
    images, image_paths = load_images(input_paths, (image_dim, image_dim))
    probs = classify_nd(model, images)
    return dict(zip(image_paths, probs))


def classify_nd(model, nd_images):
    """ Classify given a model, image array (numpy)...."""

    model_preds = model.predict(nd_images,batch_size=10000)
    # preds = np.argsort(model_preds, axis = 1).tolist()
    
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    probs = []
    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        probs.append(single_probs)
    return probs


def predict(image_source,model):
    #model = load_model()
    if image_source is None or not exists(image_source):
    	raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    image_preds = classify(model, image_source, IMAGE_DIM)
    #output_json=json.dumps(image_preds)
    return image_preds
    