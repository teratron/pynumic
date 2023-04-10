import random

import matplotlib.pyplot as plt


def main() -> None:
    # _input = 0.2
    # _hidden = 0.0
    # _output = 0.0
    target = -0.6
    neurons = [0.2, 0.0, 0.0]  # _input, _hidden, _output
    weights = [round(random.uniform(-0.5, 0.5), 3) for _ in range(2)]
    delta = [0.0, 0.0]
    print(f"{weights[0] = } {weights[1] = }")

    x: list[int] = []
    y: list[float] = []

    for i in range(20):
        print(i)
        x.append(i)

        # forward propagation
        neurons[1] = neurons[0] * weights[0]
        neurons[2] = neurons[1] * weights[1]
        print(f"{neurons[1] = :.5f} {neurons[2] = :.5f}")

        # if round(neurons[2], 3) == target:
        #     break

        # backward propagation
        delta[1] = target - neurons[2]
        y.append(round(delta[1], 3))

        # if round(neurons[2], 3) == target:
        #     break

        delta[0] = delta[1] * weights[1]
        print(f"{delta[0] = :.5f} {delta[1] = :.5f}")

        # update weights
        weights[0] += delta[0] / neurons[0]
        weights[1] += delta[1] / neurons[1]
        print(f"{weights[0] = :.5f} {weights[1] = :.5f}")

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(
            xlabel='iter',
            ylabel='miss',
            title='Loss'
    )
    ax.grid()
    # fig.savefig("test.png")
    plt.show()


if __name__ == "__main__":
    main()
