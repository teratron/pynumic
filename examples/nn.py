def main() -> None:
    _input = 0.2
    _hidden = 0.0
    _output = 0.0
    _target = 0.6
    weights = [0.1, 0.3]

    for i in range(10):
        print(i)

        _hidden = round(_input * weights[0], 5)
        _output = round(_hidden * weights[1], 5)
        print(f"{_hidden = } {_output = }")

        delta = _target - _output
        delta_weight = round(delta / _hidden, 5)
        weights[1] += delta_weight
        print(f"{delta = } {delta_weight = } {weights[1] = }")

        delta = _output - _hidden
        delta_weight = round(delta / _input, 5)
        weights[0] += delta_weight
        print(f"{delta = } {delta_weight = } {weights[0] = }")


if __name__ == "__main__":
    main()
