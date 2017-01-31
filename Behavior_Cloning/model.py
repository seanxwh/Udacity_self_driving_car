import csv
import math
import numpy as np
import random
from PIL import Image
import cv2

import tensorflow as tf
from keras.models import Model
from keras.models import model_from_json

from keras.layers import Input, Flatten, Dense, Lambda,\
                         Convolution2D, Dropout, merge
                         
from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import ELU

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


# CSV file about the where images store and the driving record stored
data_file = "./data/driving_log.csv"

# corp the top 1/3 and bottom 25 pixels from the image, then resize new 
# image to 64*64 
def preprocess_img(image, new_size_col=64,new_size_row=64):
    image = np.asarray(image)
    shape = image.shape
    image = image[math.floor(shape[0]/3):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),
                        interpolation=cv2.INTER_AREA)
    return image

# load images of different types from specificed path, and return turn 
#the images array
def imgs_preprocesses(file_paths, row):
    ary = []
    for file_dir in file_paths:
        path="./data/"+row[file_dir]
        path=path.replace(" ","")
        img = Image.open(path)
        ary.append(img)
    return ary

# brightness agumentation, for more info refer to
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
def img_brightness_argumentation(img):
    img = np.asarray(img)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .35+np.random.uniform()
    hsv_img[:,:,2] = hsv_img[:,:,2]*random_bright
    return cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB)


def img_bluring_argumentation(img):
    img = np.asarray(img)
    
    # choose a random bluring index 
    rand_blur_idx = random.choice([3,5,7])
    
    # decide whether we should use standard bluring on this step
    img = (cv2.blur(img,(rand_blur_idx,rand_blur_idx)) 
           if random.randint(0,1)==1 else img)
    
    # decide whether we should use Gaussian bluring on this step
    img = (cv2.GaussianBlur(img,(rand_blur_idx,rand_blur_idx),0) 
           if random.randint(0,1)==1 else img)
    
    # decide whether we should use median bluring on this step
    img = (cv2.medianBlur(img,rand_blur_idx) 
           if random.randint(0,1)==1 else img)
    return img


# this function is mainly used by training to generate various 
# combination of image argumentation
def img_argumentation(img):
    # decide whether we should use brightness arguementation on this step
    arg_img = (img_brightness_argumentation(img) 
               if random.randint(0,1)==1 else img)
    
    # decide whether we should use any bluring arguementation on this step
    arg_img = (img_bluring_argumentation(arg_img) 
               if random.randint(0,1)==1 else arg_img)
    
    return arg_img


def fields_generator(org_data_list, batch_size, is_training):

    while True:
        # random decide whether to shuffle the data set at the 
        # begining of a epoch 
        need_shuffle = random.randint(0,1)
        data_list = org_data_list if need_shuffle==0 else random.sample(
                                                            org_data_list,
                                                            len(org_data_list))

        # for validation, we only care how well the model predict steering 
        # angle by given original center images with corresponding speed
        if not is_training:
            print("in validation....")
            for idx, row in enumerate(data_list):
                if idx%batch_size == 0:
                    img_ary,speed_ary,steer_ary,throttle_ary = [], [], [], []
                
                [img_cn] = imgs_preprocesses(["center"], row)
                img_cn = preprocess_img(img_cn)
                img_ary.append(img_cn)
                speed_ary.append(float(row['speed']))
                steer_ary.append(float(row['steering']))
                throttle_ary.append(float(row['throttle']))
                if len(img_ary)==batch_size:
                    yield(
                    [np.array(img_ary), np.array(speed_ary)],
                    np.array(steer_ary)
                    )
                    
        # for training, we use center images only, but we do 
        # two types(flip & distortion) manipulation then feed those
        # images with corresponding speed to train the network
        else:
            print("in training....")

            for idx, row in enumerate(data_list):
                
                target_angl_data_cn = float(row['steering'])

                camera_file_type = ["center"]
                target_angl_data = [target_angl_data_cn]
                imgs = [img_cn] = imgs_preprocesses(camera_file_type,row)

                # flip image and reverse the steering angle
                fliped_img = list(map(lambda img: np.fliplr(img), imgs))
                fliped_target_angl_data = [-1*target_angl_data_cn]
                
                
                all_imgs =imgs+fliped_img
                
                all_steer_angl = target_angl_data+fliped_target_angl_data

                img_ary,speed_ary,steer_ary,throttle_ary = [], [], [], []

                for _ in range(batch_size):
                    rand_idx = random.randint(0,len(all_imgs)-1)
                    
                    img = img_argumentation(all_imgs[rand_idx])
                    img = preprocess_img(img)
                    steer_angl = all_steer_angl[rand_idx]
                    
                    #append all driving data the generator needed 
                    img_ary.append(img)
                    speed_ary.append(float(row['speed']))
                    steer_ary.append(steer_angl)
                    throttle_ary.append(float(row['throttle']))
                
                yield (
                        [np.array(img_ary), np.array(speed_ary)],
                        np.array(steer_ary)
                       )



# This model is based on the nvidia paper, with addtional speed input layer and batch normalization 
# layers for the conv layers 
# reference: 
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def model():

    images_input = Input(shape=(64, 64, 3))
    
    # normalized each pixel within an input image  
    normalized_images_input = Lambda(lambda x: x/255.0,
                                    input_shape=(64, 64, 3),
                                    output_shape=(64, 64, 3),
                                )(images_input) 
    
    model_center = Convolution2D(
                            24, 5, 5,
                            subsample=(2, 2),
                            border_mode='valid')(normalized_images_input)
    model_center = BatchNormalization()(model_center)
    model_center = ELU()(model_center)

    model_center = Convolution2D(
                            36, 5, 5,
                            subsample=(2, 2),
                            border_mode='valid')(model_center)
    model_center = BatchNormalization()(model_center)
    model_center = ELU()(model_center)

    model_center = Convolution2D(
                            48, 5, 5,
                            subsample=(2, 2),
                            border_mode='valid')(model_center)
    model_center = BatchNormalization()(model_center)
    model_center = ELU()(model_center)


    model_center = Convolution2D(
                            64, 3, 3,
                            subsample=(1, 1),
                            border_mode='valid')(model_center)
    model_center = BatchNormalization()(model_center)
    model_center = ELU()(model_center)

    model_center = Convolution2D(
                            64, 3, 3,
                            subsample=(1, 1),
                            border_mode='valid')(model_center)
    
    # flattern out the img conv layer 
    model_center = Flatten()(model_center)

    # if we want to various speed for the car, we need to input the
    # speed of the car, in order for it to perdict the right steer 
    # angle at different speed
    speed_input = Input(shape=(1,))
    # normalized the input speed with the maximun speed allowed 
    normalized_speed_input = Lambda(lambda x: x/30.19)(speed_input)
    model_speed = Dense(16)(normalized_speed_input)

    # merge the flattened img conv layer with the speed layer, and 
    # use those neuron as input to the final fully connected network
    merged = merge([model_center, model_speed], mode='concat')

    final_model = Dense(100)(merged)
    final_model = ELU()(final_model)
    # add dropout layers to prevent overfitting by introduce noise
    # https://www.quora.com/What-is-the-difference-between-Dropout-and-Batch-Normalization
    final_model = Dropout(0.4)(final_model)

    final_model = Dense(50)(merged)
    final_model = ELU()(final_model)
    # add dropout layers to prevent overfitting by introduce noise
    final_model = Dropout(0.8)(final_model)

    final_model = Dense(10)(final_model)
    final_model = ELU()(final_model)

    final_model = Dense(1)(final_model)
    
    # rescale the output by 25 times, since the output steering
    # angle can vary between -25 to 25 
    final_model = Lambda(lambda x: x*25.0)(final_model)
    
    return ([images_input, speed_input], final_model)


# this function is mainly used by continue training the network
# that has been previous trained. One can do that by setting 
# continue_training to True in main function
def load_model_and_weights(model_json, model_file):
    json_file = open(model_json,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    final_model = model_from_json(loaded_model_json)
    final_model.load_weights(model_file)
    return final_model



def main(training=True, continue_training=False):
    print("starting....")
    if training:
        copier = []

        with open(data_file, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for itm in reader:
                copier.append(itm)

        train_set = copier[::]

        vail_set = copier[-1152:][::]

        print("trainning size ", len(train_set))
        print("vailidation size ",len(vail_set))

        # if we strating a new model, receive the model from model()
        # and save the model json
        if not continue_training:
            ([img_input, speed_input], output) = model()
            final_model = Model([img_input, speed_input], output)
            model_json = final_model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
        
        # don't need to rebuild the model, but continue training the
        # existing model
        else:
            final_model = load_model_and_weights('model.json','model.h5')
        
        # initialize adam optimizer 
        adam = Adam(lr=0.0001)
        
        # we want to predict continues value rather than classification
        # hence we should use 'mse' for the loss
        final_model.compile(
                        loss='mse',
                        optimizer=adam)
        
        # save weights for each epoch, regardless of validation improvement.
        # because the validation set we created in this model is a subset of 
        # the trainning set
        checkpointer = ModelCheckpoint(
                        filepath="weights.{epoch:02d}-{val_loss:.2f}.model.h5",
                        verbose=1,
                        save_best_only=False,
                        save_weights_only=True)
        
        reduce_lr = ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.2,
                                patience=3,
                                min_lr=0.00001)

        final_model.fit_generator(
                    fields_generator(train_set, 8, True),
                    samples_per_epoch=8*len(train_set),
                    nb_epoch=20,
                    callbacks=[reduce_lr,checkpointer],
                    nb_val_samples=len(vail_set),
                    validation_data=fields_generator(vail_set,32,False),
                    max_q_size=30
                    )

        print("Saved model to disk")


if __name__ == '__main__':
    tf.app.run()
