import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

base_hyperparameters = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 300,
    'lstm_hidden_units': 8
}

learning_rate_vals = [0.0001, 0.001, 0.01]
batch_size_vals = [16, 64, 128]
lstm_hidden_units_vals = [8, 64, 256]

#dsets = {}
#
#for batch_size in batch_size_vals:
#    dsets[f'train-{batch_size}'] = tf.data.experimental.load(f'../data/dsets/train_dset-{batch_size}',
#                                                             (tf.TensorSpec(shape=(None, 300, 171), dtype=tf.float64, name=None),
#                                                              tf.TensorSpec(shape=(None, 7), dtype=tf.float32, name=None)))
#
#    dsets[f'val-{batch_size}'] = tf.data.experimental.load(f'../data/dsets/val_dset-{batch_size}',
#                                                           (tf.TensorSpec(shape=(None, 300, 171), dtype=tf.float64, name=None),
#                                                            tf.TensorSpec(shape=(None, 7), dtype=tf.float32, name=None)))
#
#    dsets[f'test-{batch_size}'] = tf.data.experimental.load(f'../data/dsets/test_dset-{batch_size}',
#                                                            (tf.TensorSpec(shape=(None, 300, 171), dtype=tf.float64, name=None),
#                                                             tf.TensorSpec(shape=(None, 7), dtype=tf.float32, name=None)))

def run_model(hyperparameters):
    lr_str = str(hyperparameters['learning_rate'])[2:]
    hyperparameter_string = f"{lr_str}-{hyperparameters['batch_size']}-{hyperparameters['epochs']}-{hyperparameters['lstm_hidden_units']}"
    print(hyperparameter_string)

    # np.random.seed(0)
    # tf.random.set_seed(0)

    # num_hidden_units = hyperparameters['lstm_hidden_units']
    # num_outputs = 7

    # model = tf.keras.Sequential([
    #         tf.keras.Input(shape=(300, 171)),
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_hidden_units,return_sequences=True)),
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_hidden_units)),
    #         tf.keras.layers.Dense(num_outputs,activation='softmax')
    # ])

    # opt = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate'])
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # train_dset = dsets[f"train-{hyperparameters['batch_size']}"]
    # val_dset = dsets[f"val-{hyperparameters['batch_size']}"]
    # es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=1)
    # history = model.fit(train_dset,epochs=hyperparameters['epochs'],validation_data = val_dset,callbacks=[es])
    # model.save(f"../models/model-{hyperparameter_string}")

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label="Train")
    # plt.plot(history.history['val_loss'], label="Validation")
    # plt.title("Loss Over Epoch")
    # plt.xlabel("Loss")
    # plt.ylabel("Epoch")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['accuracy'], label="Train")
    # plt.plot(history.history['val_accuracy'], label="Validation")
    # plt.title("Accuracy Over Epoch")
    # plt.xlabel("Accuracy")
    # plt.ylabel("Epoch")
    # plt.legend()
    # plt.show()
    # plt.savefig(f"../models/loss-accuracy-{hyperparameter_string}.png")

    # plt.close()

def test_hyperparameters():
    for learning_rate in learning_rate_vals:
        hyperparameters = copy.deepcopy(base_hyperparameters)
        hyperparameters['learning_rate'] = learning_rate
        run_model(hyperparameters)

    for batch_size in batch_size_vals:
        hyperparameters = copy.deepcopy(base_hyperparameters)
        hyperparameters['batch_size'] = batch_size
        run_model(hyperparameters)

    for lstm_hidden_units in lstm_hidden_units_vals:
        hyperparameters = copy.deepcopy(base_hyperparameters)
        hyperparameters['lstm_hidden_units'] = lstm_hidden_units
        run_model(hyperparameters)

if __name__ == '__main__':
    test_hyperparameters()

