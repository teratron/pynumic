# PyNumic

    under construction

## About

Simple neural network library for python.

## Install

```shell
$ pip install pynumic
```

## Getting Started

```python
from pynumic import PyNumic

if __name__ == '__main__':
    # Returns a new neural network
    # instance with the default parameters.
    pn = PyNumic()

    # Dataset.
    data_input  = [0.27, 0.31]
    data_target = [0.7]

    # Training dataset.
    _, _ = pn.train(data_input, data_target)
```

## Documentation

More documentation is available at the [pynumic website](https://zigenzoog.github.io/pynumic).

## Examples

You can find examples of neural networks in the 
[example's directory](https://github.com/zigenzoog/pynumic/tree/master/examples).