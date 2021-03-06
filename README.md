# Synaptic
Synaptic is a simple deep learning library for Python.

Example usage:
```python

from synaptic import NeuralNetwork
from synaptic_plus import Data, Optimizers

data = Data.load("iris-shuffled", validation_percentage=10, test_percentage=10)

nn = NeuralNetwork()

nn.training_data = data["training"]
nn.validation_data = data["validation"]

nn.hidden_layers = [100, 100]
nn.batch_size = 128
nn.optimizer = Optimizers.Adam

nn.train(epochs=5, parameters={"Learning Rate": 0.0001}, training_info=["Epoch", "Time", "Validation Accuracy"])

```
Activation Functions:
* Sigmoid
* Tanh
* Softmax
* Relu


Loss Functions:
* Logistic
* Mean Squared


Optimizers:
* Standart Gradient Descent
* Momentum
* RMSProp
* Adam
