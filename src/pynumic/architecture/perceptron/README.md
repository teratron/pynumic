# Perceptron Neural Network

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
