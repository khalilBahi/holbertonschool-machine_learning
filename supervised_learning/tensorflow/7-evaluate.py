#!/usr/bin/env python3
"""7: Evaluate"""
import tensorflow.compat.v1 as tf  # type: ignore

tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """Evaluates a neural network loaded from a TensorFlow checkpoint.

    Args:
        X (numpy.ndarray): Input data to evaluate; shape (m, nx).
        Y (numpy.ndarray): One-hot labels for X; shape (m, classes).
        save_path (str): Path prefix of the checkpoint (e.g., './model.ckpt').

    Returns:
        tuple:
            - numpy.ndarray: Network prediction y_pred; shape (m, classes).
            - float: Classification accuracy on (X, Y).
            - float: Cross-entropy loss on (X, Y).
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        y_pred_val, acc_val, loss_val = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y}
        )
    return y_pred_val, acc_val, loss_val
