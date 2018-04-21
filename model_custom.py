from keras.layers import Conv2D, BatchNormalization, LeakyReLU, add
from keras import regularizers


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

    def predict(self, features):
        pass

    def fit(self, ):
        pass



    ## GEN MODEL

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
        pass

    def create_policy_head_layer(self, x):
        pass


    def create_cnn_model(self):

        pass

    def convert_to_network_input(self, game_state):
        pass