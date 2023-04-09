def main() -> None:
    # _input = 0.2
    # _hidden = 0.0
    # _output = 0.0
    target = 0.6
    neurons = [0.2, 0.0, 0.0]  # _input, _hidden, _output
    weights = [0.1, 0.3]
    delta = [0.0, 0.0]

    for i in range(100):
        print(i)

        neurons[1] = neurons[0] * weights[0]
        neurons[2] = neurons[1] * weights[1]
        print(f"{neurons[1] = :.5f} {neurons[2] = :.5f}")

        if round(neurons[2], 3) == target:
            break

        delta[1] = target - neurons[2]
        delta[0] = delta[1] * weights[1]
        print(f"{delta[0] = :.5f} {delta[1] = :.5f}")

        weights[0] += delta[0] / neurons[0]
        weights[1] += delta[1] / neurons[1]
        print(f"{weights[0] = :.5f} {weights[1] = :.5f}")

        # _hidden = _input * weights[0]
        # _output = _hidden * weights[1]
        # print(f"{_hidden = :.5f} {_output = :.5f}")
        #
        # if round(_output, 3) == _target:
        #     break
        #
        # delta[1] = _target - _output
        # delta[0] = delta[1] * weights[1]
        # print(f"{delta[0] = :.5f} {delta[1] = :.5f}")
        #
        # weights[0] += delta[0] / _input
        # weights[1] += delta[1] / _hidden
        # print(f"{weights[0] = :.5f} {weights[1] = :.5f}")

        # delta_weight = round(delta / _hidden, 5)
        # weights[1] += delta_weight
        # print(f"{delta = } {delta_weight = } {weights[1] = }")
        #
        # delta = _output - _hidden
        # delta_weight = round(delta / _input, 5)
        # weights[0] += delta_weight
        # print(f"{delta = } {delta_weight = } {weights[0] = }")


if __name__ == "__main__":
    main()
