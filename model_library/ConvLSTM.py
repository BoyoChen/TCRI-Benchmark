import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.input_norm = layers.BatchNormalization()
        self.image_encoder_layers = [
            layers.Conv2D(filters=16, kernel_size=4, strides=2, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
            layers.BatchNormalization()
        ]
        self.rnn_block = layers.ConvLSTM2D(
            filters=64, kernel_size=4, dropout=0.0,
            recurrent_dropout=0.0, return_sequences=True
        )
        self.rnn_output_encoder = layers.Conv2D(filters=64, kernel_size=1, strides=1, activation='relu')
        self.output_layers = [
            layers.Dense(units=128, activation='relu'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=2),
            layers.Softmax()
        ]

    def apply_list_of_layers(self, input, list_of_layers, training):
        x = input
        for layer in list_of_layers:
            x = layer(x, training=training)
        return x

    def call(self, image_sequences, training):
        batch_size, encode_length, height, width, channels = image_sequences.shape

        # image_encoder block
        images = tf.reshape(
            image_sequences, [batch_size*encode_length, height, width, channels]
        )
        normalized_images = self.input_norm(images, training=training)
        encoded_images = self.apply_list_of_layers(
            normalized_images, self.image_encoder_layers, training
        )
        total_image_counts, height, width, channels = encoded_images.shape
        encoded_image_sequences = tf.reshape(
            encoded_images, [batch_size, encode_length, height, width, channels]
        )

        # rnn block
        feature_sequences = self.rnn_block(encoded_image_sequences, training=training)
        batch_size, encode_length, height, width, channels = feature_sequences.shape
        stacked_feature_sequences = tf.reshape(
            tf.transpose(feature_sequences, [0, 2, 3, 1, 4]),
            [batch_size, height, width, channels*encode_length]
        )

        # rnn_output_encoder block
        compressed_features = self.rnn_output_encoder(
            stacked_feature_sequences, training=training
        )
        flatten_feature = tf.reshape(compressed_features, [batch_size, -1])

        # output block
        output = self.apply_list_of_layers(
            flatten_feature, self.output_layers, training
        )
        return output
