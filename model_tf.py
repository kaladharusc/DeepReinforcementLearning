import tensorflow as tf
import config
from loss import softmax_cross_entropy_with_logits
import numpy as np
from keras import callbacks
from keras.models import  load_model
from settings import run_folder, run_archive_folder


class Residual_CNN_tf():
    def __init__(self, regularization_const, learning_rate, input_dimensions, output_dimensions, hidden_layers):
        self.regularization_const = regularization_const
        self.learning_rate = learning_rate
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.hidden_layers = hidden_layers
        self.no_of_layers = len(self.hidden_layers)
        self.model = self.create_model()

    def train_input_fn(self, features, labels=None, batch_size=1):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_next()

    def predict(self, x):
        return self.model.predict(input_fn=lambda : self.train_input_fn(x))


    def model_fn(self, features, labels, mode):
        # calculate value head loss
        loss_value = tf.losses.mean_squared_error(features, labels["value_head"])
        loss_policy = tf.losses.sparse_softmax_cross_entropy(labels=labels["policy_head"], logits=self.policy_head_tf)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=config.MOMENTUM
        )

        value_head_train_op = optimizer.minimize(
            loss=loss_value + loss_policy
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            estimatorSpec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_policy,
                train_op=value_head_train_op
            )
        elif mode == tf.estimator.ModeKeys.PREDICT:
            estimatorSpec = tf.estimator.EstimatorSpec(
                mode=mode,
               predictions={"classes":[],"probabilities":[]}
            )
        return estimatorSpec

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        # tbCallBack = callbacks.TensorBoard(log_dir='./Graph/2', histogram_freq=0, write_graph=True, write_images=True)


        self.model.train(
            input_fn=states
        )



        # return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split,
        #                       batch_size=batch_size, callbacks=[tbCallBack])

    def write(self, game, version):
        self.model.save(run_folder + 'models/version' + "{0:0>4}".format(version) + '.h5')

    def read(self, game, run_number, version):
        return load_model(
            run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(
                version) + '.h5',
            custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})


    def create_convolution_layer(self, x, filters, kernel):
        x = tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel,
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation=None,  # none means linear
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.regularization_const)
        )

        x = tf.layers.BatchNormalization(axis=1)(x)
        x = tf.nn.leaky_relu(x)

        return (x)

    def create_residual_layer(self, input_data, filters, kernel_dimensions):
        # with tf.variable_scope("Residual Layer"):
        x = self.create_convolution_layer(input_data, filters, kernel_dimensions)

        x = tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel_dimensions,
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation=None,  # none means linear
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.regularization_const)
        )

        x = tf.layers.BatchNormalization(axis=1)(x)

        x = tf.concat([x, input_data], axis=0)

        x = tf.nn.leaky_relu(x)

        return (x)

    def value_head(self, x):
        # with tf.variable_scope("Value Head"):
        x = tf.layers.conv2d(
            inputs=x,
            filters=1,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation=None,  # none means linear
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.regularization_const)
        )

        x = tf.layers.BatchNormalization(axis=1)(x)

        x = tf.nn.leaky_relu(x)

        x = tf.contrib.layers.flatten(x)

        x = tf.layers.dense(
            inputs=x,
            units=1,
            use_bias=False,
            activation=None,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.regularization_const)
        )

        x = tf.nn.leaky_relu(x)

        x = tf.layers.dense(inputs=x,
                            units=1,
                            use_bias=False,
                            activation=tf.tanh,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.regularization_const)
                            )

        return x

    def policy_head(self, x):
        # with tf.variable_scope("Policy Head"):

        x = tf.layers.conv2d(
            inputs=x,
            filters=2,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation=None,  # none means linear
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.regularization_const)
        )

        x = tf.layers.BatchNormalization(axis=1)(x)

        x = tf.nn.leaky_relu(x)

        x = tf.contrib.layers.flatten(x)

        x = tf.layers.dense(inputs=x,
                            units=self.output_dimensions,
                            use_bias=False,
                            activation=None,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.regularization_const),

                            )
        return (x)



    def create_model(self):
        with tf.name_scope("input"):
            # Add an op to initialize the variables.
            # init_op = tf.global_variables_initializer()

            main_input = tf.placeholder(tf.float32, shape=(None,2,3,3), name="input")
            # main_input = tf.layers.Input(
            #     shape=self.input_dimensions,
            #     batch_size=None
            # )
            x = self.create_convolution_layer(main_input,self.hidden_layers[0]["filters"],  self.hidden_layers[0]["kernel_size"])


            for hidden_layer in self.hidden_layers[1:]:
                x = self.create_residual_layer(x, hidden_layer['filters'], hidden_layer['kernel_size'])

            self.x = x
            self.value_head_tf = self.value_head(x)
            self.policy_head_tf = self.policy_head(x)


        #===================================================================================================================


            # self.session = tf.Session()
            # self.session.run(init_op)

            estimator_model = tf.estimator.Estimator(
                model_fn=self.model_fn
            )

            return estimator_model


        # ===================================================================================================================
        #     model = tf.keras.Model(inputs=[main_input], outputs=[value_head, policy_head])
        #     model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
        #                   optimizer=optimizer,
        #                   loss_weights={'value_head': 0.5, 'policy_head': 0.5}
        #                   )

            # return model


    def convertToModelInput(self, state):
        inputToModel = state.binary  # np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
        inputToModel = np.reshape(inputToModel, self.input_dimensions)
        return (inputToModel)

