import logging as lg
import config
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers, callbacks
from loss import softmax_cross_entropy_with_logits
from settings import run_folder, run_archive_folder


class CNN():
    def __init__(self, regularization_const, learning_rate, input_dimensions, output_dimensions, hidden_layers):
        self.regularization_const = regularization_const
        self.learning_rate = learning_rate
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.hidden_layers = hidden_layers
        self.no_of_layers = len(self.hidden_layers)
        self.model = self.create_cnn_model()

    ## GEN MODEL
    def predict_values(self, features):
        return self.model.predict(features)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):

        tbCallBack = callbacks.TensorBoard(log_dir='./Graph/2', histogram_freq=0, write_graph=True,
                                           write_images=True)

        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split,
                              batch_size=batch_size, callbacks=[tbCallBack])

    def write(self, game, version):
        self.model.save(run_folder + 'modells/version' + "{0:0>4".format(version) + '.h5')

    def read(self, game, run_number, version):
        return load_model(
            run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4)".format(
                version) + '.h5',
            custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

    def create_and_get_residual_layer(self, x_input, hidden_layer):
        x = self.create_and_get_conv_layer(x_input, hidden_layer=hidden_layer)

        x = Conv2D(filters=hidden_layer["filters"], kernel_size=hidden_layer["kernel_size"],
                   data_format="channels_first", padding="same", use_bias=False, activation="linear",
                   kernel_regularizer=regularizers.l2(self.regularization_const))(x)

        x = BatchNormalization(axis=1)(x)

        x = add([x_input, x])

        x = LeakyReLU()(x)

        return (x)

    def create_and_get_conv_layer(self, x, hidden_layer):
        x = Conv2D(filters=hidden_layer["filters"], kernel_size=hidden_layer["kernel_size"],
                   data_format="channels_first", padding="same", use_bias=False, activation="linear",
                   kernel_regularizer=regularizers.l2(self.regularization_const))(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return (x)

    def create_value_head_layer(self, x):
        x = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format='channels_first',
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.regularization_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(20, activation='linear', use_bias=False,
                  kernel_regularizer=regularizers.l2(self.regularization_const))(x)
        x = LeakyReLU()(x)
        x = Dense(1, activation='tanh', kernel_regularizer=regularizers.l2(self.regularization_const), use_bias=False,
                  name='name_value')(x)
        return (x)

    def create_policy_head_layer(self, x):

        x = Conv2D(
            kernel_size=(1, 1),
            data_format='channels_first',
            filters=2,
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.regularization_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(
            self.output_dimensions,
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.regularization_const),
            activation='linear',
            name='name_policy'
        )(x)

        return x

    def create_cnn_model(self):

        input_main = Input(shape=self.input_dimensions, name='name_main')
        x = self.create_and_get_conv_layer(input_main, self.hidden_layers[0])
        if len(self.hidden_layers) > 1:
            for hidden_layer in self.hidden_layers[1:]:
                x = self.create_and_get_residual_layer(x, hidden_layer)

        value_head = self.create_value_head_layer(x)
        policy_head = self.create_policy_head_layer(x)

        final_model = Model(inputs=[input_main], outputs=[value_head, policy_head])
        final_model.compile(loss={'name_value': 'mean_squared_error', 'name_policy': softmax_cross_entropy_with_logits},
                            optimizer=SGD(lr=self.learning_rate, momentum=config.MOMENTUM),
                            loss_weights={'name_value': 0.5, 'name_policy': 0.5})
        return final_model

    def convert_to_network_input(self, game_state):
        input_model = game_state.binary
        input_model = np.reshape(input_model, self.input_dimensions)
        return input_model
