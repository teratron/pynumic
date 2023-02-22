from pynumic import Pynumic

if __name__ == "__main__":
    # Returns a new neural network
    # instance with the default parameters
    # for Perceptron neural network.
    pn = Pynumic(
        bias=True,
        hidden_layers=[5, 3, 7, 4],
        activation_mode=Pynumic.TANH,
        loss_mode=Pynumic.MSE,
        loss_limit=1e-11,
        rate=0.378
    )

    print(pn.hidden_layers)
    print(pn.rate)

    # pn.query([0.1, 0.2])
    # pn.rate = 0.73
    # print(pn.rate, pn.TANH, pn.RMSE)
    # pn.calc_neurons()
    #
    # # Dataset.
    # dataset = [0.27, -0.31, -0.52, 0.66, 0.81, -0.13, 0.2, 0.49, 0.11, -0.73, 0.28]
    # len_input = 3  # Number of input data.
    # len_output = 2  # Number of output data.
    #
    # start = time.time()
    #
    # # Training.
    # len_data = len(dataset) - len_output
    # for epoch in range(1, 10001):
    #     for i in range(len_input, len_data + 1):
    #         _, _ = pn.train(dataset[i - len_input: i], dataset[i: i + len_output])
    #
    #     # Verifying.
    #     _sum = _num = 0.0
    #     for i in range(len_input, len_data + 1):
    #         _sum += pn.verify(dataset[i - len_input: i], dataset[i: i + len_output])
    #         _num += 1
    #
    #     # Average error for the entire epoch.
    #     # Exiting the cycle of learning epochs, when the minimum error level is reached.
    #     if _sum / _num < pn.loss_limit:
    #         break
    #
    # print(f"Elapsed time: {time.time() - start}")
    #
    # # Writing the neural network configuration and weights to a file.
    # pn.write(config="perceptron.json", weights="perceptron_weights.json")
    #
    # # Check the trained data, the result should be about [-0.13 0.2].
    # print(pn.query([-0.52, 0.66, 0.81]))
