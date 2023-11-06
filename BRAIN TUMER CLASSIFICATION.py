import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas

## CONVERTING THE IMAGE INTO THE ARRAY




#PREPROCESSING OF TRAINING SET
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('DATASHEET/Training',
                                                 target_size = (150, 150),
                                                 batch_size = 100,
                                                 class_mode = 'categorical')

#PREPROCESSING OF TEST SET
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('DATASHEET/Testing',
                                            target_size = (150, 150),
                                            batch_size = 100,
                                            class_mode = 'categorical')
                                                 




#building an model of VGG16

from keras.applications import vgg16

vgg=vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))

#Freeze the layers
for layers in vgg.layers:
  layers.trainable=False


from keras.src.layers.serialization import activation
from keras.src.layers import GlobalAveragePooling2D
def toplayers(Toplayer,number_class):
  Bottomlayer=Toplayer.output
  Bottomlayer=GlobalAveragePooling2D()(Bottomlayer)
  Bottomlayer=Dense(1024,activation='relu')(Bottomlayer)
  Bottomlayer=Dense(1024,activation='relu')(Bottomlayer)
  Bottomlayer=Dense(1024,activation='relu')(Bottomlayer)
  Bottomlayer=Dense(number_class,activation='softmax')(Bottomlayer)
  return Bottomlayer

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D


FC_Head = toplayers(vgg, 4)

model = Model(inputs = vgg.input, outputs = FC_Head)

print(model.summary())


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x = training_set, validation_data = test_set, epochs = 25)

#Predicting the result
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('DATASHEET/predict/m_2.jpg',target_size=(150,150))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)

result=model.predict(test_image)
print(result)

training_set.class_indices

if(result[0][0]==1):
    print('glioma')
elif(result[0][1]==1):
    print('meningioma')
elif(result[0][2]==1):
    print('notumor')
else:
    print('pituitary')
        
        
