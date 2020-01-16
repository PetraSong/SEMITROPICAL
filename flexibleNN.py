#!/hpc/local/CentOS7/dhl_ec/software/slideLearn/anaconda3/bin/ python3
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization, Concatenate
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.models import Sequential
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
import argparse
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Convolutional neural network for the analasys of atherosclerotic plaques')
parser.add_argument('--base_dir',metavar='IMAGEBASE',help='base directory for the input files: top folder', required=True)
parser.add_argument('--weight_path',metavar='weights',help='enter the desired path + filename for your weight file (.h5 extention)',required=True)
parser.add_argument('--amount_train',metavar='train',help='enter the amount of train images', required=True)
parser.add_argument('--amount_test', metavar='test_and_val',help='enter the amount of test and validation images (assuming they are the same)',required = True)
parser.add_argument('--confu',metavar='confu',help='enter a path for the confusion matrix', required=False, default="confu_out.png")
parser.add_argument('--classes',metavar='class_list',help='please enter your classes in order of bin (0, 1, 2 etc.) separated by spaces: 0 1 2 or: event no_event', required=True,nargs='+')
args = parser.parse_args()

base_model = applications.InceptionV3(weights=None,
                                include_top=False,
                                input_shape=(None, None,3))
base_model.trainable = False

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#globals

#paths; fit these to the correct sets of images
trainpath=args.base_dir+"/train"
testpath=args.base_dir+"/test"
valpath=args.base_dir+"/val"

#if you have a dataset of uniform shape, None can be changed to said shape, please note that this also makes the use of globalaveragepooling no longer mandatory
shapex=None
shapey=None
channels=3

#amount of classes in the classification
num_classes=len(args.classes)

#amount of images in a batch, higher is better, but memory might not allow it
batch_size=40

#amount of files to be trained on, the actual amount of files multiplied by a number of choice (possible because of image augmentation
#train files = 771 each class
numfilestrain=int(args.amount_train)*6
numfilestest=int(args.amount_test)*5

#the (true) amount of files in the validation set
numfilesval=args.amount_test

#amount of iterations of the network (steps per epoch is equal to number of images divided by batch size)
epochs=15

#names of the subfolders your images are in; classnames
dirlist=args.classes


#model design

add_model = Sequential()
#add_model.add(base_model)
add_model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(None, None, 3)))
add_model.add(MaxPooling2D(pool_size=(2, 2)))

#this layer may be disabled if a uniform shape is used
add_model.add(GlobalAveragePooling2D())

add_model.add(Dropout(0.05))
add_model.add(Dense(50,activation='relu'))
add_model.add(Dense(100,activation='relu'))
add_model.add(Dense(50,activation='relu'))
add_model.add(Dense(num_classes,
                    activation='softmax'))
CNN = add_model

CNN.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#image augmentation settings; can be edited to be more 'extreme'
traindatagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.15, height_shift_range=0.15, shear_range=0.15,
	horizontal_flip=True, fill_mode="constant")
testdatagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.15, height_shift_range=0.15, shear_range=0.15,
	horizontal_flip=True, fill_mode="constant")

#data generator for the validation set keeps every image as is
validationgen = ImageDataGenerator(fill_mode="constant")
validation_generator = validationgen.flow_from_directory(valpath, color_mode='rgb', batch_size=batch_size, shuffle=False, save_format='png', interpolation='nearest',class_mode=None)

#early stopper: stops the model if no improvements are shown for patience=x epochs
earlystoppersymptomsma = EarlyStopping(monitor='val_acc', patience=8, verbose=1)

#saving the model so it can be used later
checkpointersymptomsma = ModelCheckpoint(args.weight_path
                                ,monitor='val_acc'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=True)

H = CNN.fit_generator(traindatagen.flow_from_directory(trainpath,
							   color_mode='rgb', batch_size=batch_size, shuffle=True, save_format='png',
							   interpolation='nearest',class_mode="categorical"), validation_data=testdatagen.flow_from_directory(testpath,
							   color_mode='rgb', batch_size=batch_size, shuffle=True, save_format='png', interpolation='nearest',class_mode="categorical"),
							   steps_per_epoch=numfilestrain/batch_size,epochs=epochs,validation_steps=numfilestest/batch_size,callbacks=[earlystoppersymptomsma, checkpointersymptomsma])
CNN.load_weights(args.weight_path)

#some small amount of data generation
predicted = CNN.predict_generator(validation_generator, numfilesval/batch_size)
print(predicted)
pred  = np.argmax(predicted, axis=1)

print(confusion_matrix(validation_generator.classes, pred))
binary = confusion_matrix(validation_generator.classes,pred)
target_names = dirlist
print(classification_report(validation_generator.classes, pred, target_names=target_names))

#plotting and saving a confusion matrix
fig,ax = plot_confusion_matrix(conf_mat=binary,class_names=target_names)
plt.savefig(args.confu, bbox_inches='tight')
