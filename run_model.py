#!/usr/bin/python
# -*- coding: utf-8 -*-

import h5py
import keras
import numpy as np
import json

from model import get_model

class ValAccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_accs = []

    def on_epoch_end(self, batch, logs={}):
        self.val_accs.append(logs.get('val_acc'))

# Determine the number of epoch for training
training_epochs = 200
# Determine batch size
batch_size = 32

# Import data from the h5 file
train_image_data = []
train_audio_data = []
train_labels = []
dataset = h5py.File("./data/AVLetters_data.h5", "r")
for name in ["Nicola", "Anya", "Steve", "Kate", "Yi", "Bill", "John", "Stephen", "Faye", "Verity"]:
    train_image_data.append(np.array(dataset.get("%s/images" % name))[:, :40, :, :, :])
    train_audio_data.append(np.array(dataset.get("%s/audio" % name))[:, :40, :])
    train_labels.append(np.array(dataset.get("%s/one_hot" % name)))
train_image_data = np.concatenate(train_image_data, axis=0)
train_audio_data = np.concatenate(train_audio_data, axis=0)
train_one_hot = np.concatenate(train_labels, axis=0)

# Dictionary made to gather and organize results
json_content_file = {"Nicola": {}, "Anya": {}, "Steve": {}, "Kate": {}, "Yi": {}, "Bill": {}, "John": {}, "Stephen": {}, "Faye": {}, "Verity": {}}

for repetition in range(1, 11):
    # Perform the Leave One Out Cross Validation
    for idx, name in enumerate(["Nicola", "Anya", "Steve", "Kate", "Yi", "Bill", "John", "Stephen", "Faye", "Verity"]):
        start = 78 * idx
        end = (idx + 1) * 78
        validation_image_data = train_image_data[start:end, :, :, :, :]
        validation_audio_data = train_audio_data[start:end, :, :]
        validation_labels = train_one_hot[start:end, :]

        training_image_data = np.concatenate(
            [train_image_data[:start, :, :, :, :], train_image_data[end:, :, :, :, :]], axis=0)
        training_audio_data = np.concatenate([train_audio_data[:start, :, :], train_audio_data[end:, :, :]],
                                             axis=0)
        training_labels = np.concatenate([train_one_hot[:start, :], train_one_hot[end:, :]], axis=0)

        # Import the model
        model = get_model()
        # Build the model with the given parameters
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Create a callback to get validation accuracy
        history = ValAccHistory()

        # Train the model with given hyperparameters
        model.fit([training_image_data, training_audio_data], training_labels, batch_size=batch_size, epochs=training_epochs,
                  validation_data=([validation_image_data, validation_audio_data], validation_labels),
                  callbacks=[history],
                  verbose=1)

        # Store the history composed of evaluation on the test data
        json_content_file[name][repetition] = history.val_accs

with open("results_XFlow.json", "w") as fil:
    json.dump(json_content_file, fil)
