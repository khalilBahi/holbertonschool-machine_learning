#!/usr/bin/env python3
""" Task 6: 6. Train """
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train: numpy.ndarray, training input data.
        Y_train: numpy.ndarray, training labels.
        X_valid: numpy.ndarray, validation input data.
        Y_valid: numpy.ndarray, validation labels.
        layer_sizes: list of int, number of nodes in each layer of the network.
        activations: list of activation functionsfor each layer of the network.
        alpha: float, the learning rate.
        iterations: int, the number of iterations to train over.
        save_path: str, path to save the trained model.

    Returns:
        str: The path where the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add tensors and operations to the graph's collection
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            feed_train = {x: X_train, y: Y_train}
            feed_valid = {x: X_valid, y: Y_valid}

            if i % 100 == 0 or i == iterations:
                train_cost, train_accuracy = sess.run(
                    [loss, accuracy], feed_dict=feed_train)
                valid_cost, valid_accuracy = sess.run(
                    [loss, accuracy], feed_dict=feed_valid)
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

            sess.run(train_op, feed_dict=feed_train)

        save_path = saver.save(sess, save_path)
        print(f"Model saved in path: {save_path}")

    return save_path
