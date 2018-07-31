# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:16:20 2018

@author: mavinaya
"""
from keras.applications import VGG16
from keras.applications import models
from keras.applications import layers
from keras import optimizers
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from os import listdir
from pickle import dump
import numpy as np


directory = 'D:/Users/Default User/Desktop/ML/car-damage-dataset'
base_dir = 'D:/Users/Default User/Desktop/ML'
top_model_weights_path = 'bottleneck_fc_model.h5'
img_width  = 256
img_height = 256

train_samples = [len(listdir(directory+'/data1/training/'+i)) for i in sorted(listdir(directory+'/data1/training/'))]
nb_train_samples = sum(train_samples)

validation_samples = [len(listdir(directory+'/data1/validation/'+i)) for i in sorted(listdir(directory+'/data1/validation/'))]
nb_validation_samples = sum(validation_samples)


# extract features from each photo in the directory
def extract_features(directory):
    # load the model
    in_layer = Input(shape=(256, 256, 3))
    model = VGG16(include_top=False, input_tensor=in_layer)
    print(model.summary())

    # load the images and extract the features
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))        
        
        # prepare the image for the VGG model
        image = preprocess_input(image)
        
        # get features
        feature = model.predict(image, verbose=0)
        
        # get image id
        image_id = name.split('.')[0]
        
        # store feature
        features[image_id] = feature
#        print(features)
        
        # save features
    dump(features, open('features.pkl', 'wb'))
        

def save_bottleneck_features(directory):
    print('save_bottleneck_features..')
    datagen = ImageDataGenerator(rescale=1./255)
    model = VGG16(include_top=False, weights='imagenet')

    train_data_location = directory + '/data1/training/'    
    print(train_data_location)
    train_generator = datagen.flow_from_directory(train_data_location,
                                            target_size=(img_width, img_height),
                                            batch_size=16,
                                            class_mode=None)

    print('point 1..')
    train_features = model.predict_generator(train_generator, nb_train_samples)
    np.save(open('D:\\Users\\Default User\\Desktop\\ML\\train_features.npy', 'wb'), train_features)

    print('point 2..')
    validation_data_location = directory + '/data1/validation/'
    validation_generator = datagen.flow_from_directory(validation_data_location,
                                            target_size=(img_width, img_height),
                                            batch_size=16,
                                            class_mode=None)
    print('point 3..')
    validation_features = model.predict_generator(validation_generator, nb_validation_samples)
    np.save(open('D:\\Users\\Default User\\Desktop\\ML\\validation_features.npy', 'wb'), validation_features)
    print('save_bottleneck_features end..')       


def build_model():
     print('build_model..')
    
     train_features = base_dir + '/train_features.npy'
        
     train_data = np.load(open(train_features, 'rb'))
     train_labels = np.array([0]*1000 + [1] * 1000)

     validation_features = base_dir + '/validation_features.npy'
     validation_data = np.load(open(validation_features, 'rb'))
     validation_lables = np.array([0]*400 + [1] * 400)
     
#    model = VGG16(include_top=False, weights='imagenet')
     model = models.Sequential()
     model.add(layers.Flatten(input_shape=train_data.shape[1:]))
     model.add(layers.Dense(256, activation='relu'))
     model.add(layers.Dropout(0.5))
     model.add(layers.Dense(1, activation='sigmoid'))
     
     model.compile(optimizer=optimizers.RMSprop,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
            
     model.fit(train_data, train_labels,
               epochs=50,
               batch_size=16,
               validation_data=(validation_data, validation_lables))
     model.save_weights('D:\\Users\\Default User\\Desktop\\ML\\bottleneck_fc_model.h5')
     print('build_model end..')
     return model


save_bottleneck_features(directory)
build_model()

    
#if __name__ == '__main__':
#    main()
