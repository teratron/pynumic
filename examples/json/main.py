"""TODO:"""

import os

from pynumic import Pynumic

if __name__ == "__main__":
    # Returns a new neural network instance from config.
    pn = Pynumic(os.path.join("config", "perceptron.json"))

    # Dataset.
    data_input = [1.0, 1.0]
    data_target = [0.0]

    # Training dataset.
    _, _ = pn.train(data_input, data_target)

    # Writing weights to a file.
    pn.write(weights="perceptron_weights.json")
