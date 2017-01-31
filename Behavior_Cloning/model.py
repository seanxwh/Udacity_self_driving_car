import csv
import math
import numpy as np
from copy import deepcopy
# np.random.seed(1987)
import scipy
import random
# random.seed(1987)
from random import shuffle
from PIL import Image
import json
import cv2
from scipy.ndimage.interpolation import zoom
from skimage import transform



import tensorflow as tf
# tf.set_random_seed(1987)
from keras.layers import Input, Flatten, Dense
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge,merge, Lambda
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.callbacks import ReduceLROnPlateau



data_file = "./data/driving_log.csv"


def preprocess_img(image, new_size_col=64,new_size_row=64):
    image = np.asarray(image)
    shape = image.shape
    image = image[math.floor(shape[0]/3):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),
                        interpolation=cv2.INTER_AREA)
    return image

# resize data
def imgs_preprocesses(file_paths, row):
    ary = []
    for file_dir in file_paths:
        path="./data/"+row[file_dir]
        path=path.replace(" ","")
        img = Image.open(path)
        ary.append(img)
    return ary


def img_brightness_argumentation(img):
    img = np.asarray(img)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .35+np.random.uniform()
    hsv_img[:,:,2] = hsv_img[:,:,2]*random_bright
    return cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB)


def img_bluring_argumentation(img):
    img = np.asarray(img)
    rand_blur_idx = random.choice([3,5,7])
    img = cv2.blur(img,(rand_blur_idx,rand_blur_idx)) if random.randint(0,1)==1 else img
    img = cv2.GaussianBlur(img,(rand_blur_idx,rand_blur_idx),0) if random.randint(0,1)==1 else img
    img = cv2.medianBlur(img,rand_blur_idx) if random.randint(0,1)==1 else img
    return img


def img_argumentation(img):
    arg_img = img_brightness_argumentation(img) if random.randint(0,1)==1 else img
    arg_img = img_bluring_argumentation(arg_img) if random.randint(0,1)==1 else arg_img
    return arg_img


def fields_generator(org_data_list, batch_size, is_training):

    train_data_fields = ["center","left","right","steering","throttle","brake","speed"]


    while True:
        need_shuffle = random.randint(0,1)
        data_list = org_data_list if need_shuffle==0 else random.sample(org_data_list,len(org_data_list))
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

        else:
            print("in training....")

            for idx, row in enumerate(data_list):

                camera_file_type = ["center","left","right"]
                target_angl_data_cn = float(row['steering'])

                camera_file_type = ["center"]
                target_angl_data = [target_angl_data_cn]
                imgs = [img_cn] = imgs_preprocesses(camera_file_type,row)

                fliped_imgs = list(map(lambda img: np.fliplr(img), imgs))
                fliped_target_angl_data = [-1*target_angl_data_cn]

                all_imgs =imgs+fliped_imgs
                #
                all_steer_angl = target_angl_data+fliped_target_angl_data

                img_ary,speed_ary,steer_ary,throttle_ary = [], [], [], []

                for _ in range(batch_size):
                    rand_idx = random.randint(0,len(all_imgs)-1)
                    img = img_argumentation(deepcopy(all_imgs[rand_idx]))
                    img = preprocess_img(img)
                    steer_angl = all_steer_angl[rand_idx]
                    img_ary.append(img)
                    speed_ary.append(float(row['speed']))
                    steer_ary.append(steer_angl)
                    throttle_ary.append(float(row['throttle']))

                yield (
                        [np.array(img_ary), np.array(speed_ary)],
                        np.array(steer_ary)
                       )




def model():

    images_input = Input(shape=(64, 64, 3))

    normalized_images_input=Lambda(lambda x: x/255.0,
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


    model_center = Flatten()(model_center)

    speed_input = Input(shape=(1,))
    normalized_speed_input = Lambda(lambda x: x/30.19)(speed_input)
    model_speed = Dense(16)(normalized_speed_input)


    merged = merge([model_center, model_speed], mode='concat')


    final_model = Dense(100)(merged)
    final_model = ELU()(final_model)
    final_model = Dropout(0.4)(final_model)

    final_model = Dense(50)(merged)
    final_model = ELU()(final_model)
    final_model = Dropout(0.8)(final_model)


    final_model = Dense(10)(final_model)
    final_model = ELU()(final_model)

    final_model = Dense(1)(final_model)
    final_model = Lambda(lambda x: x*25.0)(final_model)

    return ([images_input, speed_input], final_model)


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
        target_field = "steering"

        copier = []

        with open(data_file, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for itm in reader:
                copier.append(itm)

        train_set = copier[::]

        vail_set = copier[-1152:][::]

        print("trainning size ", len(train_set))
        print("vailidation size ",len(vail_set))


        if not continue_training:
            ([img_input, speed_input], output) = model()
            final_model = Model([img_input, speed_input], output)

            model_json = final_model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            final_model.save_weights('model.h5')
        else:
            final_model = load_model_and_weights('model.json','model.h5')

        rmsprop = RMSprop(lr=0.0005)
        adam = Adam(lr=0.0001)

        final_model.compile(
                        loss='mse',
                        optimizer=adam)

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
                    nb_val_samples = len(vail_set),
                    validation_data = fields_generator(vail_set,32,False),
                    max_q_size=30
                    )

        print("Saved model to disk")


if __name__ == '__main__':
    tf.app.run()
