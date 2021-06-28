import numpy as np
import tensorflow as tf
from collections import defaultdict
from modules.training_helper import calculate_metric_dict


def train(
    model,
    datasets,
    summary_writer,
    saving_path,
    max_epoch,
    evaluate_freq,
    class_weight,
    learning_rate
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))
    class_weight, norm = tf.linalg.normalize(tf.cast(class_weight, tf.float32), ord=1)

    @tf.function
    def train_step(model, image_sequences, RI_labels):
        with tf.GradientTape() as tape:
            model_output = model(image_sequences, training=True)
            sample_weight = tf.reduce_sum(RI_labels*class_weight, axis=-1)
            batch_loss = loss_function(RI_labels, model_output, sample_weight=sample_weight)

        gradients = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        avg_losses['cross entropy'].update_state(batch_loss)
        return

    best_prauc = 0
    best_heidke = -np.inf
    for epoch_index in range(1, max_epoch+1):
        print(f'Executing epoch #{epoch_index}')

        for image_sequences, RI_labels, frame_ID_ascii in datasets['train']:
            train_step(model, image_sequences, RI_labels)

        with summary_writer['train'].as_default():
            for loss_name, avg_loss in avg_losses.items():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch_index)
                avg_loss.reset_states()

        if epoch_index % evaluate_freq == 0:
            print(f'Completed {epoch_index} epochs, do some evaluation')

            for phase in ['train', 'test', 'valid']:
                metric_dict = calculate_metric_dict(model, datasets[phase])
                with summary_writer[phase].as_default():
                    for metric_name, metric_value in metric_dict.items():
                        tf.summary.scalar(metric_name, metric_value, step=epoch_index)

            valid_prauc = metric_dict['prauc']
            valid_heidke = metric_dict['heidke']
            if best_prauc < valid_prauc:
                best_prauc = valid_prauc
                model.save_weights(saving_path/'best-prauc', save_format='tf')
            if best_heidke < valid_heidke:
                best_heidke = valid_heidke
                model.save_weights(saving_path/'best-heidke', save_format='tf')
