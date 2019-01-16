#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Conv2DTranspose, AvgPool2D
from keras.layers import add, concatenate, Input, Lambda
from keras.layers import LSTM, TimeDistributed, Masking
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization


# Constants for the AVletters dataset
number_max_frames = 40
input_CNN_shape = (number_max_frames, 60, 80, 1)
input_MLP_shape = (number_max_frames, 26, )
number_classes = 26


def get_model():
    # Build the CNN - pre-cross-connections
    input_CNN = Input(shape=input_CNN_shape, name='InputCNN')
    input_norm = TimeDistributed(Flatten())(input_CNN)
    input_norm = Masking(mask_value=0.)(input_norm)
    input_norm = TimeDistributed(Reshape((60, 80, 1)))(input_norm)
    input_norm = BatchNormalization(axis=1, name='bn_CNN_input')(input_norm)

    # Original XFlow
    visual_convolution_1 = TimeDistributed(Convolution2D(8, (3, 3), border_mode='same', activation='relu'), name='conv11')(input_norm)
    # Configuration with PReLU
    # visual_convolution_1 = TimeDistributed(Convolution2D(8, (3, 3), border_mode='same', activation=PReLU()), name='conv11')(input_norm)

    # Original XFlow
    pooling_1 = TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name='maxpool1')(visual_convolution_1)
    # Configuration with Avg pooling
    # pooling_1 = TimeDistributed(AvgPool2D((2,2), strides=(2, 2)), name='avgpool1')(visual_convolution_1)

    # Build the MLP - pre-cross-connections
    input_MLP = Input(shape=input_MLP_shape, name='input_MLP')
    input_masked = Masking(mask_value=0., input_shape=input_MLP_shape)(input_MLP)

    # Original XFlow
    hidden_layer_MLP = TimeDistributed(Dense(32, activation='relu'), name='fc1')(input_masked)
    # Configuration with PReLU
    # hidden_layer_MLP = TimeDistributed(Dense(32, activation=PReLU()), name='fc1')(input_masked)

    # Add the 1st cross modal connection - CNN to MLP (comment this part if you want to test the architecture without cross connections)
    cross_CNN_to_MLP_1 = TimeDistributed(Convolution2D(8, (1, 1), border_mode='same'))(pooling_1)
    cross_CNN_to_MLP_1 = TimeDistributed(PReLU())(cross_CNN_to_MLP_1)
    cross_CNN_to_MLP_1 = TimeDistributed(Flatten())(cross_CNN_to_MLP_1)
    cross_CNN_to_MLP_1 = TimeDistributed(Dense(32))(cross_CNN_to_MLP_1)
    cross_CNN_to_MLP_1 = TimeDistributed(PReLU())(cross_CNN_to_MLP_1)

    # Add 1st cross residual connection - from CNN input to MLP (comment this part if you want to test the architecture without cross residual connections)
    # Original XFlow
    cross_residual_CNN_to_MLP_1 = TimeDistributed(MaxPooling2D((4,4), strides=(4,4)))(input_norm)
    # Configuration with Avg pooling
    # cross_residual_CNN_to_MLP_1 = TimeDistributed(AvgPool2D((4,4), strides=(4,4)))(input_norm)
    cross_residual_CNN_to_MLP_1 = TimeDistributed(Flatten())(cross_residual_CNN_to_MLP_1)
    cross_residual_CNN_to_MLP_1 = TimeDistributed(Dense(32))(cross_residual_CNN_to_MLP_1)
    cross_residual_CNN_to_MLP_1 = TimeDistributed(PReLU())(cross_residual_CNN_to_MLP_1)

    # Add the 1st cross modal connection - MLP to CNN (comment this part if you want to test the architecture without cross connections)
    cross_MLP_to_CNN_1 = TimeDistributed(Dense(25*15))(hidden_layer_MLP)
    cross_MLP_to_CNN_1 = TimeDistributed(PReLU())(cross_MLP_to_CNN_1)
    cross_MLP_to_CNN_1 = TimeDistributed(Reshape((15,25,1)))(cross_MLP_to_CNN_1)
    cross_MLP_to_CNN_1 = TimeDistributed(Conv2DTranspose(8, (16, 16), padding='valid'))(cross_MLP_to_CNN_1)
    cross_MLP_to_CNN_1 = TimeDistributed(PReLU())(cross_MLP_to_CNN_1)

    # Add 1st cross residual connection - from MLP input to CNN (comment this part if you want to test the architecture without cross connections)
    cross_residual_MLP_to_CNN_1 = TimeDistributed(Dense(25*15))(input_masked)
    cross_residual_MLP_to_CNN_1 = TimeDistributed(PReLU())(cross_residual_MLP_to_CNN_1)
    cross_residual_MLP_to_CNN_1 = TimeDistributed(Reshape((15,25,1)))(cross_residual_MLP_to_CNN_1)
    cross_residual_MLP_to_CNN_1 = TimeDistributed(Conv2DTranspose(8, (16, 16), padding='valid'))(cross_residual_MLP_to_CNN_1)
    cross_residual_MLP_to_CNN_1 = TimeDistributed(PReLU())(cross_residual_MLP_to_CNN_1)

    # CNN - post-cross-connections 1
    post_pooling_1 = add([pooling_1, cross_residual_MLP_to_CNN_1])
    # Comment the line above and uncomment the line below when trying without cross residual connections
    # post_pooling_1 = pooling_1
    visual_post_connection_1 = concatenate([post_pooling_1, cross_MLP_to_CNN_1])
    # Comment the line above and uncomment the line below when trying without cross modal connections
    # visual_post_connection_1 = post_pooling_1

    # Original XFlow
    visual_convolution_2 = TimeDistributed(Convolution2D(16, (3, 3), border_mode='same', activation='relu'), name='conv21')(visual_post_connection_1)
    # Configuration with PReLU
    # visual_convolution_2 = TimeDistributed(Convolution2D(16, (3, 3), border_mode='same', activation=PReLU()), name='conv21')(visual_post_connection_1)

    # Original XFlow
    pooling_2 = TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name='maxpool2')(visual_convolution_2)
    # Configuration with Avg pooling
    # pooling_2 = TimeDistributed(AvgPool2D((2,2), strides=(2, 2)), name='avgpool2')(visual_convolution_2)


    # MLP - post-cross-connections 1
    post_hidden_layer = add([hidden_layer_MLP, cross_residual_CNN_to_MLP_1])
    # Comment the line above and uncomment the line below when trying without cross residual connections
    # post_hidden_layer = hidden_layer_MLP
    audio_post_connection_1 = concatenate([post_hidden_layer, cross_CNN_to_MLP_1])
    # Comment the line above and uncomment the line below when trying without cross modal connections
    # audio_post_connection_1 = post_hidden_layer

    # Original XFlow
    output_layer_MLP = TimeDistributed(Dense(32, activation='relu'), name='fc2')(audio_post_connection_1)
    # Configuration with PReLU
    # output_layer_MLP = TimeDistributed(Dense(32, activation=PReLU()), name='fc2')(audio_post_connection)

    # Add the 2nd cross modal connection - CNN to MLP (comment this part if you want to test the architecture without cross connections)
    cross_CNN_to_MLP_2 = TimeDistributed(Convolution2D(16, (1, 1), border_mode='same'))(pooling_2)
    cross_CNN_to_MLP_2 = TimeDistributed(PReLU())(cross_CNN_to_MLP_2)
    cross_CNN_to_MLP_2 = TimeDistributed(Flatten())(cross_CNN_to_MLP_2)
    cross_CNN_to_MLP_2 = TimeDistributed(Dense(64))(cross_CNN_to_MLP_2)
    cross_CNN_to_MLP_2 = TimeDistributed(PReLU())(cross_CNN_to_MLP_2)

    # Add 2nd cross residual connection from CNN input to MLP (comment this part if you want to test the architecture without cross residual connections)
    # Original XFlow
    cross_residual_CNN_to_MLP_2 = TimeDistributed(MaxPooling2D((8,8), strides=(4,8)))(input_norm)
    # Configuration with Avg pooling
    # cross_residual_CNN_to_MLP_2 = TimeDistributed(AvgPool2D((8,8), strides=(4,8)))(input_norm)
    cross_residual_CNN_to_MLP_2 = TimeDistributed(Flatten())(cross_residual_CNN_to_MLP_2)
    cross_residual_CNN_to_MLP_2 = TimeDistributed(Dense(32))(cross_residual_CNN_to_MLP_2)
    cross_residual_CNN_to_MLP_2 = TimeDistributed(PReLU())(cross_residual_CNN_to_MLP_2)

    # Cross-connections - MLP to CNN (comment this part if you want to test the architecture without cross connections)
    cross_MLP_to_CNN_2 = TimeDistributed(Dense(13*8))(output_layer_MLP)
    cross_MLP_to_CNN_2 = TimeDistributed(PReLU())(cross_MLP_to_CNN_2)
    cross_MLP_to_CNN_2 = TimeDistributed(Reshape((8,13,1)))(cross_MLP_to_CNN_2)
    cross_MLP_to_CNN_2 = TimeDistributed(Conv2DTranspose(16, (8, 8), padding='valid'))(cross_MLP_to_CNN_2)
    cross_MLP_to_CNN_2 = TimeDistributed(PReLU())(cross_MLP_to_CNN_2)

    # 2nd cross residual connection from MLP input to CNN (comment this part if you want to test the architecture without cross residual connections)
    cross_residual_MLP_to_CNN_2 = TimeDistributed(Dense(13*8))(input_masked)
    cross_residual_MLP_to_CNN_2 = TimeDistributed(PReLU())(cross_residual_MLP_to_CNN_2)
    cross_residual_MLP_to_CNN_2 = TimeDistributed(Reshape((8,13,1)))(cross_residual_MLP_to_CNN_2)
    cross_residual_MLP_to_CNN_2 = TimeDistributed(Conv2DTranspose(16, (8, 8), padding='valid'))(cross_residual_MLP_to_CNN_2)
    cross_residual_MLP_to_CNN_2 = TimeDistributed(PReLU())(cross_residual_MLP_to_CNN_2)

    # CNN - post-cross-connections 2
    post_pooling_2 = add([pooling_2, cross_residual_MLP_to_CNN_2])
    # Comment the line above and uncomment the line below when trying without cross residual connections
    # post_pooling_2 = pooling_2
    visual_post_connection_2 = concatenate([post_pooling_2, cross_MLP_to_CNN_2])
    # Comment the line above and uncomment the line below when trying without cross modal connections
    # visual_post_connection_2 = post_pooling_2

    visual_reshape = TimeDistributed(Flatten(), name='flatten1')(visual_post_connection_2)
    # Original XFlow
    fully_connected_CNN = TimeDistributed(Dense(64, activation='relu'), name='fcCNN')(visual_reshape)
    # Configuration with PReLU
    # fcCNN = TimeDistributed(Dense(64, activation=PReLU()), name='fcCNN')(reshape)

    # Concatenate the models
    audio_post_connection_2 = add([output_layer_MLP, cross_residual_CNN_to_MLP_2])
    # Comment the line above and uncomment the line below when trying without cross residual connections
    # audio_post_connection_2 = output_layer_MLP
    shared_representation = concatenate([fully_connected_CNN, audio_post_connection_2, cross_CNN_to_MLP_2])
    # Comment the line above and uncomment the line below when trying without cross modal connections
    # shared_representation = concatenate([fully_connected_CNN, audio_post_connection_2])
    normalized_shared_representation = BatchNormalization(axis=1, name='mergebn')(shared_representation)
    input_LSTM = Dropout(0.5, name='mergedrop')(normalized_shared_representation)

    lstm = LSTM(64)(input_LSTM)
    output_model = Dense(number_classes, activation='softmax')(lstm)

    # Return the model object
    model = Model(input=[input_CNN, input_MLP], output=output_model)
    return model
