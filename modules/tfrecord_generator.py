import tensorflow as tf
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from modules.data_downloader import download_data


def remove_outlier_and_nan(numpy_array, upper_bound=1000):
    numpy_array = np.nan_to_num(numpy_array, copy=False)
    numpy_array[numpy_array > upper_bound] = 0
    VIS = numpy_array[:, :, :, 2]
    VIS[VIS > 1] = 1  # VIS channel ranged from 0 to 1
    return numpy_array


def flip_SH_images(image_matrix, info_df):
    SH_idx = info_df.index[info_df.region == 'SH']
    image_matrix[SH_idx] = np.flip(image_matrix[SH_idx], 1)
    return image_matrix


def data_cleaning_and_organizing(image_matrix, info_df):
    image_matrix = remove_outlier_and_nan(image_matrix)
    image_matrix = flip_SH_images(image_matrix, info_df)
    return image_matrix, info_df


def data_split(image_matrix, info_df, phase):
    if phase == 'train':
        target_index = info_df.index[info_df.ID < '2015000']
    elif phase == 'valid':
        target_index = info_df.index[np.logical_and('2017000' > info_df.ID, info_df.ID > '2015000')]
    elif phase == 'test':
        target_index = info_df.index[info_df.ID > '2017000']

    new_image_matrix = image_matrix[target_index]
    new_info_df = info_df.loc[target_index].reset_index(drop=True)
    return new_image_matrix, new_info_df


def group_by_id(image_matrix, info_df):

    id2indices_group = info_df.groupby('ID', sort=False).groups
    indices_groups = list(id2indices_group.values())

    image_matrix = [image_matrix[indices] for indices in indices_groups]
    info_df = [info_df.iloc[indices] for indices in indices_groups]

    return image_matrix, info_df


def write_tfrecord(image_matrix, info_df, tfrecord_path):

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _encode_tfexample(single_TC_images, single_TC_info):
        history_len = single_TC_info.shape[0]
        frame_ID = single_TC_info.ID + '_' + single_TC_info.time
        features = {
            'history_len': _int64_feature(history_len),
            'images': _bytes_feature(np.ndarray.tobytes(single_TC_images)),
            'intensity': _bytes_feature(np.ndarray.tobytes(single_TC_info.Vmax.to_numpy())),
            'frame_ID': _bytes_feature(np.ndarray.tobytes(frame_ID.to_numpy('bytes')))
        }
        return tf.train.Example(features=tf.train.Features(feature=features))

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        assert(len(image_matrix) == len(info_df))
        for single_TC_images, single_TC_info in zip(image_matrix, info_df):
            example = _encode_tfexample(single_TC_images, single_TC_info)
            serialized = example.SerializeToString()
            writer.write(serialized)


def generate_tfrecord(data_folder):
    file_path = Path(data_folder, 'TCSA.h5')
    if not file_path.exists():
        print(f'file {file_path} not found! try to download it!')
        download_data(data_folder)
    with h5py.File(file_path, 'r') as hf:
        image_matrix = hf['images'][:]
    # collect info from every file in the list
    info_df = pd.read_hdf(file_path, key='info', mode='r')
    image_matrix, info_df = data_cleaning_and_organizing(image_matrix, info_df)

    phase_data = {
        phase: data_split(image_matrix, info_df, phase)
        for phase in ['train', 'valid', 'test']
    }
    del image_matrix, info_df

    for phase, (image_matrix, info_df) in phase_data.items():
        image_matrix, info_df = group_by_id(image_matrix, info_df)
        phase_path = Path(data_folder, f'TCSA.tfrecord.{phase}')
        write_tfrecord(image_matrix, info_df, phase_path)


def get_or_generate_tfrecord(data_folder):
    tfrecord_path = {}
    for phase in ['train', 'valid', 'test']:
        phase_path = Path(data_folder, f'TCSA.tfrecord.{phase}')
        if not phase_path.exists():
            print(f'tfrecord {phase_path} not found! try to generate it!')
            generate_tfrecord(data_folder)
        tfrecord_path[phase] = phase_path

    return tfrecord_path
