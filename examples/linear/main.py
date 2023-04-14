"""TODO:"""
from pynumic import Pynumic


def main() -> None:
    """TODO:"""
    # Returns a new neural network
    # instance with the default parameters,
    # same n = nn.New("perceptron").
    pn = Pynumic()

    # Properties.
    pn(bias=True, hidden_layers=[3, 2])
    pn.activation_mode = pn.LINEAR
    pn.loss_mode = pn.MSE
    pn.loss_limit = 1e-5  # .0001

    # Dataset that doesn't need to be scaled.
    data_input = [10.6, -5.0, 200.0]
    data_target = [5.0, -50.3]

    # Training dataset.
    print("Train:", *pn.train(data_input, data_target))

    # Check the trained data, the result should be about [5 -50.3].
    print("Check:", pn.query(data_input))

    # pn.write("config.json", flag="config")
    # pn.write(config="config.json", weights="weights.json")
    # print(pn.__dict__)


if __name__ == "__main__":
    main()
