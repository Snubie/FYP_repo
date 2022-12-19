# Import necessary modules
import tensorflow as tf

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob
import random

def cnn_model():

    inputs = Input(shape=(600, 400, 3))

    conv1 = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(pool3)

    up1 = Conv2D(filters=32, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(conv4))
    merge1 = concatenate([conv3, up1], axis = 3)

    up2 = Conv2D(filters=16, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(merge1))
    merge2 = concatenate([conv2, up2], axis = 3)

    up3 = Conv2D(filters=8, kernel_size=(2,2), activation = 'relu', padding = 'same')(UpSampling2D()(merge2))
    merge3 = concatenate([conv1, up3], axis = 3)

    outputs = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation='relu')(merge3)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train(config):

    # training dataset 
    train_df = pd.DataFrame({'lowlight': glob.glob(config.lowlight_train_images_path)})
    test_df = pd.DataFrame({ 'normal': glob.glob(config.result_train_images_path)})

    data_gen_args = dict(
        rescale= 1./255,  # Normalize pixel values
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_datagen = ImageDataGenerator(**data_gen_args)
    test_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = random.randint(1, 1000000)
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col = 'lowlight',
        target_size = (600, 400),  # Resize images to a fixed size
        batch_size = config.train_batch_size,
        class_mode = None,
        seed = seed
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='normal',
        target_size=(600, 400),  # Resize images to a fixed size
        batch_size=config.train_batch_size,
        class_mode = None,
        seed = seed
    )

    #Compile the model
    model = cnn_model()
    # print(model.summary())
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(), 
        metrics = ["accuracy"])

    generator = zip(train_generator, test_generator)

    model.fit(
        generator,
        steps_per_epoch = (train_df.size / config.train_batch_size),
        epochs = config.epochs,
        batch_size= config.train_batch_size
    )

    # for e in range(model.epochs):
    # print('Epoch: ', e)
    # batches = 0
    # for x_batch, y_batch in generator.flow(x_train, y_train, batch_size=32):
    #     model.fit(x_batch, y_batch)
    #     batches += 1
    #     if batches >= len(x_train) / 32:
    #         # we need to break the loop by hand because
    #         # the generator loops indefinitely
    #         break

    model.save(config.model_loc)

def test(config):
    model = load_model(config.model_loc)

    low_df = pd.DataFrame({'low': glob.glob(config.lowlight_test_images_path)})
    normal_df = pd.DataFrame({ 'normal': glob.glob(config.result_test_images_path)})

    data_gen_args = dict(rescale= 1./255)

    train_datagen = ImageDataGenerator(**data_gen_args)
    test_datagen = ImageDataGenerator(**data_gen_args)
    seed = random.randint(1, 1000000)

    train_generator = train_datagen.flow_from_dataframe(
        low_df,
        x_col = 'low',
        target_size = (600, 400),  # Resize images to a fixed size
        batch_size = config.train_batch_size,
        class_mode = None,
        seed = seed
    )

    test_generator = test_datagen.flow_from_dataframe(
        normal_df,
        x_col='normal',
        target_size=(600, 400),  # Resize images to a fixed size
        batch_size=config.train_batch_size,
        class_mode = None,
        seed = seed
    )

    print(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_train_images_path', type=str, default=".\\LOL\\Real_captured\\Train\\Low\\*.png")
    parser.add_argument('--result_train_images_path', type=str, default=".\\LOL\\Real_captured\\Train\\Normal\\*.png")
    parser.add_argument('--lowlight_test_images_path', type=str, default=".\\LOL\\Real_captured\\Test\\Low\\*.png")
    parser.add_argument('--result_test_images_path', type=str, default=".\\LOL\\Real_captured\\Test\\Normal\\*.png")

    parser.add_argument('--model_loc', type=str, default=".\\Model\\")

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=32)

    parser.add_argument('--total_test_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=50)

    config = parser.parse_known_args()[0]
    train(config)


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


