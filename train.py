# Import necessary modules
import tensorflow as tf

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
import glob
import random

def ms_ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))


def cnn_model():

    inputs = Input(shape=(600, 400, 3))

    # conv1 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    # conv1 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    # conv2 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(pool1)
    # conv2 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    # conv3 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(pool2)
    # conv3 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    # conv4 = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(pool3)
    # conv4 = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(conv4)

    # up1 = Conv2D(filters=64, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv4))
    # merge1 = concatenate([conv3, up1], axis = 3)
    # conv5 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(merge1)
    # conv5 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(conv5)

    # up2 = Conv2D(filters=32, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv5))
    # merge2 = concatenate([conv2, up2], axis = 3)
    # conv6 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(merge2)
    # conv6 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv6)

    # up3 = Conv2D(filters=16, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv6))
    # merge3 = concatenate([conv1, up3], axis = 3)
    # conv7 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(merge3)
    # conv7 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(conv7)

    
    conv1 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(pool3)
    conv4 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(conv4)

    up1 = Conv2D(filters=32, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv4))
    merge1 = concatenate([conv3, up1], axis = 3)
    conv5 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(merge1)
    conv5 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv5)

    up2 = Conv2D(filters=16, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv5))
    merge2 = concatenate([conv2, up2], axis = 3)
    conv6 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(merge2)
    conv6 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(conv6)

    up3 = Conv2D(filters=8, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv6))
    merge3 = concatenate([conv1, up3], axis = 3)
    conv7 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(merge3)
    conv7 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(conv7)

    outputs = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation='relu')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train(config):

    # training dataset 
    train_df = pd.DataFrame({ "lowlight" : glob.glob(config.lowlight_train_images_path)})
    test_df = pd.DataFrame({ "normal" : glob.glob(config.result_train_images_path)})

    data_gen_args = dict(
        rescale= 1./255,  # Normalize pixel values
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode= "nearest"
    )

    train_datagen = ImageDataGenerator(**data_gen_args)
    test_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = random.randint(1, 1000000)
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col = "lowlight",
        target_size = (600, 400),  # Resize images to a fixed size
        batch_size = config.train_batch_size,
        class_mode = None,
        seed = seed
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col="normal",
        target_size=(600, 400),  # Resize images to a fixed size
        batch_size=config.train_batch_size,
        class_mode = None,
        seed = seed
    )

    #Compile the model
    model = cnn_model()
    # print(model.summary())
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss = ms_ssim_loss, 
        metrics = [ms_ssim_loss, "accuracy"])

    generator = zip(train_generator, test_generator)

    model.fit(
        generator,
        steps_per_epoch = (train_df.size / config.train_batch_size),
        epochs = config.epochs,
        batch_size= config.train_batch_size
    )

    model.save(config.model_loc)

    # TODO: Add preprocessing layer after complete building model

def test(config):
    model = load_model(config.model_loc, custom_objects={'ms_ssim_loss': ms_ssim_loss})

    low_ds = image_dataset_from_directory(
        directory= config.lowlight_test_images_path,
        labels=None,
        label_mode=None,
        batch_size=48,
        image_size=(600, 400))

    normal_ds = image_dataset_from_directory(
        directory= config.result_test_images_path,
        labels=None,
        label_mode=None,
        batch_size=48,
        image_size=(600, 400))

    low_ds = low_ds.map(lambda x: (tf.divide(x, 255.0)))
    normal_ds = normal_ds.map(lambda x: (tf.divide(x, 255.0)))

    # for low, normal in zip(low_ds, normal_ds)

    result = model.evaluate(zip(low_ds, normal_ds), batch_size = 48, verbose = 0)
    print(model.metrics_names)
    print(result)

def predict(config):
    model = load_model(config.model_loc, custom_objects={'ms_ssim_loss': ms_ssim_loss})

    for filename in os.listdir(config.predict_images_input_path):
        image_bgr = cv2.imread(os.path.join(config.predict_images_input_path,filename))

        if image_bgr is not None:
            image_bgr = cv2.resize(image_bgr, (600,400), interpolation = cv2.INTER_AREA)  
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = image_rgb / 255.0
            image_rgb = np.array([image_rgb])
            image_rgb = np.transpose(image_rgb,(0,2,1,3))
            
            result = model.predict(image_rgb)

            result = np.transpose(result,(0,2,1,3))
            result = result * 255.0
            result = cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(config.predict_images_output_path, filename), result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_train_images_path', type=str, default="./LOL/Real_captured/Train/Low/*.png")
    parser.add_argument('--result_train_images_path', type=str, default="./LOL/Real_captured/Train/Normal/*.png")

    parser.add_argument('--lowlight_test_images_path', type=str, default="./LOL/Real_captured/Test/Low")
    parser.add_argument('--result_test_images_path', type=str, default="./LOL/Real_captured/Test/Normal")

    parser.add_argument('--predict_images_input_path', type=str, default="./Predict/Input/")
    parser.add_argument('--predict_images_output_path', type=str, default="./Predict/Output/")

    parser.add_argument('--model_loc', type=str, default="./Model/")

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=32)

    parser.add_argument('--total_test_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=50)

    config = parser.parse_known_args()[0]
    predict(config)


# parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--weight_decay', type=float, default=0.0001)
# parser.add_argument('--grad_clip_norm', type=float, default=0.1)

# parser.add_argument('--val_batch_size', type=int, default=8)
# parser.add_argument('--num_workers', type=int, default=4)
# parser.add_argument('--display_iter', type=int, default=10)
# parser.add_argument('--snapshot_iter', type=int, default=10)
# parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
# parser.add_argument('--load_pretrain', type=bool, default= False)
# parser.add_argument('--pretrain_dir', type=str, default= "./snapshots/Epoch99.pth")

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Set the memory limit for the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(f'Successfully created {len(logical_gpus)} Logical GPU(s)')
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

# plt.figure()

# #subplot(r,c) provide the no. of rows and columns
# f, axarr = plt.subplots(2,2) 

# # use the created array to output your multiple images. In this case I have stacked 4 images vertically
# axarr[0][0].imshow(X_batch[0])
# axarr[0][1].imshow(X_batch[1])
# axarr[1][0].imshow(Y_batch[0])
# axarr[1][1].imshow(Y_batch[1])

# plt.show()


