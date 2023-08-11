"""TODO:"""
import matplotlib.pyplot as plt

from pynumic import Pynumic


def main() -> None:
    """TODO:"""
    # Returns a new neural network
    # instance with the default parameters.
    pn = Pynumic()

    # Properties.
    pn(bias=True, hidden_layers=[3, 2], rate=0.3)
    pn.activation_mode = pn.LINEAR
    pn.loss_mode = pn.RMSE
    pn.loss_limit = 1e-5

    # Dataset that doesn't need to be scaled.
    data_input = [10.6, -5.0, 200.0]
    data_target = [5.0, -50.3]

    # Training dataset.
    print("Train:", *pn.train(data_input, data_target))

    # Check the trained data, the result should be about [5 -50.3].
    print("Check:", pn.query(data_input))

    pn.write("config.json", flag="config")
    # pn.write(config="config.json", weights="weights.json")
    # print(pn.__dict__)

    _fig, ax = plt.subplots()
    length = len(pn.data.count)
    ax.plot(pn.data.count[:length], pn.data.loss[:length])
    ax.set(
        xlabel='iter',
        ylabel='loss',
        title='Loss'
    )
    ax.grid()
    # fig.savefig("test.png")
    plt.show()


if __name__ == "__main__":
    main()
