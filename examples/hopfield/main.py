from pynumic import PyNumic

if __name__ == "__main__":
    # Returns a new neural network
    # instance with the default parameters
    # for Hopfield neural network.

    pn = PyNumic("hopfield", energy=0.8)
    pn.energy = 0.23
    print(pn, pn.energy)
