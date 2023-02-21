# PyNumic

    under construction

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pynumic)](https://pypi.org/project/pynumic)
[![PyPI - Version](https://img.shields.io/pypi/v/pynumic.svg)](https://pypi.org/project/pynumic)
[![PyPI - Downloads](https://img.shields.io/pypi/dd/pynumic)](https://pypi.org/project/pynumic/#files)
[![License](https://img.shields.io/github/license/zigenzoog/pynumic)](https://pypi.org/project/pynumic)
[![Code style: docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba)](https://github.com/PyCQA/docformatter)
[![Code style: mypy](https://img.shields.io/badge/code%20style-mypy-green)](https://github.com/python/mypy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

___

## Description

Simple neural network library for python.

## Visuals

    Depending on what you are making, it can be a good idea to include screenshots or even a video\
    (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help,\
    but check out Asciinema for a more sophisticated method.

## Installation

```shell
$ pip install pynumic
```

## Usage

```python
from pynumic import Pynumic

if __name__ == '__main__':
    # Returns a new neural network
    # instance with the default parameters.
    pn = Pynumic()

    # Dataset.
    data_input  = [0.27, 0.31]
    data_target = [0.7]

    # Training dataset.
    _, _ = pn.train(data_input, data_target)
```

## Documentation

### Properties of Perceptron Neural Network

### _name_

Neural network architecture name (required field for a config).

### _bias_

The neuron bias, false or true (required field for a config).

### _hidden_layer_

Array of the number of neurons in each hidden layer.

### _activation_mode_

ActivationMode function mode (required field for a config).

| Code | Activation | Description                              |
|------|------------|------------------------------------------|
| 0    | LINEAR     | Linear/identity                          |
| 1    | RELU       | ReLu (rectified linear unit)             |
| 2    | LEAKYRELU  | Leaky ReLu (leaky rectified linear unit) |
| 3    | SIGMOID    | Logistic, a.k.a. sigmoid or soft step    |
| 4    | TANH       | TanH (hyperbolic tangent)                |

### _loss_mode_

The mode of calculation of the total error.

| Code | Loss   | Description             |
|------|--------|-------------------------|
| 0    | MSE    | Mean Squared Error      |
| 1    | RMSE   | Root Mean Squared Error |
| 2    | ARCTAN | Arctan                  |
| 3    | AVG    | Average                 |

### _loss_limit_

Minimum (sufficient) limit of the average of the error during training.

### _rate_

Learning coefficient (greater than 0.0 and less than or equal to 1.0).

More documentation is available at the [pynumic website](https://zigenzoog.github.io/pynumic).

---

## Examples

- [perceptron](examples/perceptron)
- [linear](examples/linear)
- [query](examples/query)
- [and_train](examples/and_train)
- [json](examples/json)

You can find examples of neural networks in the
[example's directory](examples).

## Support

    Tell people where they can go to for help.\
    It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap

    If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing

    State if you are open to contributions and what your requirements are for accepting them.

    For people who want to make changes to your project,\
    it's helpful to have some documentation on how to get started.\
    Perhaps there is a script that they should run or some environment variables that they need to set.\
    Make these steps explicit.\
    These instructions could also be useful to your future self.

    You can also document commands to lint the code or run tests.\
    These steps help to ensure high code quality and reduce the likelihood\
    that the changes inadvertently break something.\
    Having instructions for running tests is especially helpful if it requires external setup,\
    such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment

    Show your appreciation to those who have contributed to the project.

## License

[MIT License](LICENSE).

## Project status

Project at the initial stage.

See the latest commits on [https://github.com/teratron/pynumic](https://github.com/teratron/pynumic)

---

![My Skills](https://skillicons.dev/icons?i=py,golang,javascript,react,html,css,sass,git,github)
