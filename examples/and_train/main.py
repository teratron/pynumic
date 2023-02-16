from pynumic import Pynumic

props = {
    "name": "perceptron",
    "bias": True,
    "activation_mode": 3,
    "loss_mode": 0,
    "loss_limit": 1e-10,
    "weights": [
        [
            [-2.5128086, 2.6974556, 3.034397, -2.4341068],
            [-1.2436904, -1.1729956, 4.4001436, -2.1053333],
            [-1.2884712, 2.5544305, 10.886107, -2.1163273],
            [3.9765725, 3.646633, 4.741202, -3.8852577],
            [8.725591, 3.0480642, 3.0672483, -7.115494],
        ],
        [
            [3.7148979, 2.9444046, -5.72786, 2.2840204, -1.6592604, 0.33781952],
            [1.8408697, 2.070344, -6.0672054, 3.9654624, -2.7668004, 2.3363395],
            [1.8098677, 2.2063692, 0.08325871, -4.959725, 5.3901534, 1.0965135],
        ],
        [
            [2.1007898, 6.552546, -5.262143, -1.1054513],
            [-6.4693666, -4.019415, -3.8858104, 6.2537074],
        ],
    ],
}

if __name__ == "__main__":
    # Returns a new neural network instance from config.
    pn = Pynumic(**props)
    # pn = Pynumic(**{"name": "perceptron"})

    # Dataset:
    data_input = [0.27, 0.31, 0.52]  # input dataset.
    data_target = [0.7, 0.1]  # target dataset.

    # Getting the results of the trained network.
    data_output = pn.query(data_input)
    print(data_output)

    # If there is target data, then we can train the received output data.
    count, loss = pn.and_train(data_target)
    print(count, loss)

    # Check the trained data, the result should be about [0.7 0.1].
    print(pn.query(data_input))
