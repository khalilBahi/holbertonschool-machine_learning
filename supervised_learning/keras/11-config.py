#!/usr/bin/env python3
""" Task 11: 11. Save and Load Configuration """
import tensorflow.keras as K  # type: ignore


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Args:
        network: The model whose configuration should be saved.
        filename: The path of the file that the
        configuration should be saved to.

    Returns:
        None
    """
    json_string = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_string)
    return None


def load_config(filename):
    """
    Loads a model with a specific configuration.

    Args:
        filename: The path of the file containing
        the model's configuration in JSON format.

    Returns:
        The loaded model.
    """
    with open(filename, "r") as f:
        network_string = f.read()
    network = K.models.model_from_json(network_string)
    return network
