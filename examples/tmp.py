from pynumic import Pynumic


def main() -> None:
    pn = Pynumic(weights=[2.3])
    # pn = Pynumic()
    print(pn.weights)

    # pn(weights=[2.3])
    pn(bias=True, rate=0.91, activation_mode=1)
    print(pn.weights)


if __name__ == "__main__":
    main()
