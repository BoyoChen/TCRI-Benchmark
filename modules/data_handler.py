import tensorflow as tf
from functools import partial
import tensorflow_addons as tfa
from modules.tfrecord_generator import get_or_generate_tfrecord


def ascii_array_to_string(ascii_array):
    string = ''
    for ascii_code in ascii_array:
        string += chr(ascii_code)
    return string


def deserialize(serialized_TC_history):
    features = {
        'history_len': tf.io.FixedLenFeature([], tf.int64),
        'images': tf.io.FixedLenFeature([], tf.string),
        'intensity': tf.io.FixedLenFeature([], tf.string),
        'frame_ID': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(serialized_TC_history, features)
    history_len = tf.cast(example['history_len'], tf.int32)

    images = tf.reshape(
        tf.io.decode_raw(example['images'], tf.float32),
        [history_len, 128, 128, 4]
    )
    intensity = tf.reshape(
        tf.io.decode_raw(example['intensity'], tf.float64),
        [history_len]
    )
    intensity = tf.cast(intensity, tf.float32)

    frame_ID_ascii = tf.reshape(
        tf.io.decode_raw(example['frame_ID'], tf.uint8),
        [history_len, -1]
    )

    return images, intensity, history_len, frame_ID_ascii


def breakdown_into_sequence(
    images, intensity, history_len, frame_ID_ascii, encode_length, estimate_distance
):
    sequence_num = history_len - (encode_length + estimate_distance) + 1
    starting_index = tf.range(sequence_num)

    image_sequences = tf.map_fn(
        lambda start: images[start: start+encode_length],
        starting_index, fn_output_signature=tf.float32
    )

    starting_frame_ID_ascii = frame_ID_ascii[encode_length - 1:-estimate_distance]
    starting_intensity = intensity[encode_length - 1:-estimate_distance]
    ending_intensity = intensity[encode_length + estimate_distance - 1:]
    intensity_change = ending_intensity - starting_intensity

    RI_threshold = 35
    is_RI = (intensity_change >= RI_threshold)
    is_not_RI = tf.math.logical_not(is_RI)
    RI_labels = tf.stack([is_not_RI, is_RI], axis=-1)
    RI_labels = tf.cast(RI_labels, tf.float32)

    return tf.data.Dataset.from_tensor_slices(
        (image_sequences, RI_labels, starting_frame_ID_ascii)
    )


def image_preprocessing(images, intensity, history_len, frame_ID_ascii, rotate_type):
    two_channel_images = tf.gather(images, [0, 3], axis=-1)
    if rotate_type == 'single':
        angles = tf.random.uniform([history_len], maxval=360)
        rotated_images = tfa.image.rotate(two_channel_images, angles=angles)
    elif rotate_type == 'series':
        angles = tf.ones([history_len]) * tf.random.uniform([1], maxval=360)
        rotated_images = tfa.image.rotate(two_channel_images, angles=angles)
    else:
        rotated_images = two_channel_images
    images_64x64 = tf.image.central_crop(rotated_images, 0.5)

    return images_64x64, intensity, history_len, frame_ID_ascii


def get_tensorflow_datasets(
    data_folder, batch_size, encode_length,
    estimate_distance, rotate_type
):
    tfrecord_paths = get_or_generate_tfrecord(data_folder)
    datasets = dict()
    for phase, record_path in tfrecord_paths.items():
        serialized_TC_histories = tf.data.TFRecordDataset(
            [record_path], num_parallel_reads=8
        )
        TC_histories = serialized_TC_histories.map(
            deserialize, num_parallel_calls=tf.data.AUTOTUNE
        )

        min_history_len = encode_length + estimate_distance
        long_enough_histories = TC_histories.filter(
            lambda x, y, l, n: l >= min_history_len
        )

        preprocessed_histories = long_enough_histories.map(
            partial(
                image_preprocessing,
                rotate_type=rotate_type
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        TC_sequence = preprocessed_histories.interleave(
            partial(
                breakdown_into_sequence,
                encode_length=encode_length,
                estimate_distance=estimate_distance
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = TC_sequence.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(4)
        datasets[phase] = dataset

    return datasets
