"""
@author: Maneesha Vinayak
"""

import json
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Sequential, load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint, History

location = 'path/to/data'
train_data_dir = location+'/path/to/training/data'
validation_data_dir = location+'/path/to/validation/data'

# expected image size
img_width, img_height = 256, 256
# how many images to be considered for training
train_samples = 1840
# how many images to be used for validation
validation_samples = 460

num_epochs = 20

top_model_weights_path = location+'/model_weights.h5'
fine_tuned_model_path = location+'/trained_model.h5'

#Transfer learning with VGG16 
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
## set model architecture 
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu', W_regularizer=l2(0.01)))
add_model.add(Dropout(0.5))
#use activation softmax for multiple categories
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
#can use categorical_crossentropy depending on the output classes
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')


# All images will be rescaled by 1./255
datagen = ImageDataGenerator(rescale=1./255)
      
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                target_size=(img_width, img_height),
                                batch_size=16, 
                                class_mode='binary', 
                                shuffle=False)

validation_generator = datagen.flow_from_directory(validation_data_dir,
                                target_size=(img_width, img_height),
                                batch_size=16,
                                class_mode='binary',
                                shuffle=False)

checkpoint = ModelCheckpoint(fine_tuned_model_path, monitor='val_acc', 
                                verbose=1, save_best_only=True, 
                                save_weights_only=False, mode='auto')

# fine-tune the model
trained_model = model.fit_generator(train_generator, steps_per_epoch=100, epochs=num_epochs, 
                  validation_data=validation_generator, validation_steps=50,callbacks=[checkpoint])

model.save(location+'/trained_model.h5')

model_json_final = model.to_json()
with open(location+'/ft_history.txt', 'wb') as f:
          json.dump(trained_model.history, f)
        
with open(location+'/ft_history.json', 'w') as f:
          f.write(trained_model_json_final)
        
print("Model saved")
