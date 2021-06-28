import tensorflow as tf


def calculate_metric_dict(model, dataset):
    PR_AUC = tf.keras.metrics.AUC(curve='PR')
    TP = tf.keras.metrics.TruePositives(thresholds=0.5)
    TN = tf.keras.metrics.TrueNegatives(thresholds=0.5)
    FP = tf.keras.metrics.FalsePositives(thresholds=0.5)
    FN = tf.keras.metrics.FalseNegatives(thresholds=0.5)
    ACC = tf.keras.metrics.Accuracy()

    for image_sequences, RI_labels, frame_ID_ascii in dataset:
        pred_prob = model(image_sequences, training=False)
        is_RI = RI_labels[:, 1]
        pred_RI_prob = pred_prob[:, 1]
        pred_is_RI = tf.cast(pred_RI_prob > 0.5, tf.float32)

        PR_AUC.update_state(is_RI, pred_RI_prob)
        TP.update_state(is_RI, pred_RI_prob)
        TN.update_state(is_RI, pred_RI_prob)
        FP.update_state(is_RI, pred_RI_prob)
        FN.update_state(is_RI, pred_RI_prob)
        ACC.update_state(is_RI, pred_is_RI)

    tp = TP.result()
    tn = TN.result()
    fp = FP.result()
    fn = FN.result()
    percision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1_score = 2 * percision * recall / (percision + recall)
    heidke = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    acc = ACC.result()
    prauc = PR_AUC.result()

    return dict(
        percision=percision,
        recall=recall,
        f1_score=f1_score,
        heidke=heidke,
        acc=acc,
        prauc=prauc
    )
