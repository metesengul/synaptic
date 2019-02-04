import numpy as np
from synaptic_plus import \
    ActivationFunctions,\
    LossFunctions,\
    InitializationFunctions,\
    Optimizers,\
    Data

import time


class NeuralNetwork:

    def __init__(self):

        # Get & Set
        self._weights = []
        self._biases = []

        self._hidden_layers = []

        self._initialization_function = InitializationFunctions.relu_optimized
        self._loss_function = LossFunctions.mean_squared
        self._optimizer = Optimizers.Standart
        self._hidden_activation_functions = ActivationFunctions.relu
        self._output_activation_function = ActivationFunctions.sigmoid
        self._hidden_dropouts = 0

        self._batch_size = None

        self._training_data = {}
        self._validation_data = {}
        self._test_data = {}

        # Get
        self._layers = []

        self._training_info = []

        self._dw = []
        self._db = []

        # None
        self._activation_functions = []
        self._dropouts = []

        self._feature_length = 0
        self._label_length = 0

        self._training_data_batch = {}

        self._z_cache = []

    @property
    def layers(self):
        return tuple(self._layers)

    @property
    def initialization_function(self):
        return self._initialization_function

    @initialization_function.setter
    def initialization_function(self, value):
        self._initialization_function = value

    @property
    def output_activation_function(self):
        return self._output_activation_function

    @output_activation_function.setter
    def output_activation_function(self, value):
        self._output_activation_function = value

    @property
    def hidden_activation_functions(self):
        return self._hidden_activation_functions

    @hidden_activation_functions.setter
    def hidden_activation_functions(self, value):
        self._hidden_activation_functions = value

    @property
    def loss_function(self):
        return self._loss_function

    @ loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    @property
    def training_info(self):
        return self._training_info

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, value):
        self._biases = value

    @property
    def dw(self):
        return self._dw

    @property
    def db(self):
        return self._db

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def hidden_layers(self):
        return tuple(self._hidden_layers)

    @hidden_layers.setter
    def hidden_layers(self, value):
        if not self._layers:
            self._hidden_layers = value
        else:
            raise Exception("You can only set layers once. Please create another object with different layers.")

    @property
    def hidden_dropouts(self):
        return self._hidden_dropouts

    @hidden_dropouts.setter
    def hidden_dropouts(self, value):
        self._hidden_dropouts = value

    def _build(self):
        if not self._layers:
            self._build_layers()
            self._build_weights_and_biases()
        self._build_deltas()
        self._build_z_cache()
        self._build_activation_functions()
        self._build_dropouts()

    def _build_layers(self):
        if self._feature_length == 0 or self._label_length == 0:
            raise Exception("Please set training data first.")
        else:
            self._layers += [self._feature_length]
            self._layers += self._hidden_layers
            self._layers += [self._label_length]

    def _build_weights_and_biases(self):
        # Create Weights
        for i in range(len(self._layers) - 1):
            self._weights += [(np.random.rand(self._layers[i + 1], self._layers[i]) - 0.5) *
                              self._initialization_function(i, self._layers)]

        # Create Biases
        for i in range(len(self._layers) - 1):
            self._biases += [(np.random.rand(self._layers[i + 1], 1) - 0.5) *
                             self._initialization_function(i, self._layers)]

    def _build_z_cache(self):
        if self._batch_size is None:
            self._batch_size = self._training_data["length"]
        for i in range(len(self._layers)):
            self._z_cache += [np.ones(shape=(self._layers[i], self._batch_size))]

    def _build_deltas(self):
        # Setup of Deltas
        for i in range(self._optimizer.space + 1):
            self._dw += [[]]
            self._db += [[]]

        # Create Deltas
        for j in range(self._optimizer.space + 1):

            for i in range(len(self._layers) - 1):
                self._dw[j] += [np.zeros(shape=(self._layers[i + 1], self._layers[i]))]

            for i in range(len(self._layers) - 1):
                self._db[j] += [np.zeros(shape=(self._layers[i + 1], 1))]

    def _build_activation_functions(self):
        self._activation_functions = [ActivationFunctions.linear]
        if type(self._hidden_activation_functions) != list and type(self._hidden_activation_functions) != tuple:
            for i in range(len(self._layers) - 1):
                if i != len(self._layers) - 2:
                    self._activation_functions += [self._hidden_activation_functions]
                else:
                    self._activation_functions += [self._output_activation_function]
        else:
            self._activation_functions += self._hidden_activation_functions
            self._activation_functions += [self._output_activation_function]

    def _build_dropouts(self):
        self._dropouts = [0]
        if type(self._hidden_dropouts) != list and type(self._hidden_dropouts) != tuple:
            if len(self._layers) > 2:
                for i in range(len(self._layers) - 2):
                    self._dropouts += [self._hidden_dropouts]
        else:
            self._dropouts += self._hidden_dropouts
        self._dropouts += [0]

    @property
    def training_data(self):
        if self._training_data is {}:
            return None
        features = np.transpose(self._training_data["features"])
        labels = np.transpose(self._training_data["labels"])
        data = {"features": features, "labels": labels}
        return data

    @property
    def validation_data(self):
        if self._validation_data is {}:
            return None
        features = np.transpose(self._validation_data["features"])
        labels = np.transpose(self._validation_data["labels"])
        data = {"features": features, "labels": labels}
        return data

    @property
    def test_data(self):
        if self._test_data is {}:
            return None
        features = np.transpose(self._test_data["features"])
        labels = np.transpose(self._test_data["labels"])
        data = {"features": features, "labels": labels}
        return data

    @training_data.setter
    def training_data(self, data):
        training_data = data
        self._training_data["length"] = len(training_data["features"])
        self._training_data["features"] = np.transpose(training_data["features"])
        self._training_data["labels"] = np.transpose(training_data["labels"])

        self._feature_length = len(training_data["features"][0])
        self._label_length = len(training_data["labels"][0])

    @validation_data.setter
    def validation_data(self, data):
        validation_data = data
        self._validation_data["length"] = len(validation_data["features"])
        self._validation_data["features"] = np.transpose(validation_data["features"])
        self._validation_data["labels"] = np.transpose(validation_data["labels"])

    @test_data.setter
    def test_data(self, data):
        test_data = data
        self._test_data["length"] = len(test_data["features"])
        self._test_data["features"] = np.transpose(test_data["features"])
        self._test_data["labels"] = np.transpose(test_data["labels"])

    def _feed_forward(self, inputs):

        z = inputs
        self._z_cache[0] = z

        for i in range(len(self._weights)):

            z = np.add(np.dot(self._weights[i], z), self._biases[i])
            self._z_cache[i + 1] = z
            z = self._activation_functions[i + 1](z)

        return z

    def _backpropagate(self):

        self._feed_forward(self._training_data_batch["features"])

        da = self._loss_function(
            self._activation_functions[-1](self._z_cache[-1]), self._training_data_batch["labels"], derivative=True)

        for i in reversed(range(len(self._weights))):

            dz = self._activation_functions[i + 1](self._z_cache[i + 1], derivative=True)

            dadz = da * dz

            a = self._activation_functions[i](self._z_cache[i])

            # Dropout Algorithm
            dropout_prob = self._dropouts[i]
            if dropout_prob > 0:
                dropout_vector = np.random.rand(a.shape[0], a.shape[1]) > dropout_prob
                a = np.multiply(a, dropout_vector)
                a = np.divide(a,  1 - dropout_prob)

            dw = np.dot(dadz, np.transpose(a)) / self._batch_size

            db = np.sum(dadz, axis=1, keepdims=True) / self._batch_size

            self._dw[0][i] = dw
            self._db[0][i] = db

            da = np.dot(np.transpose(self._weights[i]), dadz)

    def train(self, epochs, parameters, training_info=("epoch", "training_cost")):

        self._build()

        parameters = Data.query(parameters)
        training_info = Data.query(training_info)

        self._training_info = []
        self._add_training_info(training_info, 0, "NA")

        # Calculate loop time (According to batch size and training data size)
        batch_loop = 1
        if self._batch_size != self._training_data["length"]:
            batch_loop = self._training_data["length"] - (self._training_data["length"] % self._batch_size)
            batch_loop /= self._batch_size
            batch_loop = int(batch_loop)

        for j in range(epochs):

            start_time = time.time()

            for b in range(batch_loop):

                # Set Mini Batch
                self._training_data_batch["features"] = \
                    self._training_data["features"][:, b * self._batch_size:(b + 1) * self._batch_size]

                self._training_data_batch["labels"] = \
                    self._training_data["labels"][:, b * self._batch_size:(b + 1) * self._batch_size]

                # Backpropagate
                self._backpropagate()

                # Optimize
                self._optimizer.run(parameters, self)

            elapsed_time = np.round(time.time() - start_time, 3)

            self._add_training_info(training_info, j + 1, elapsed_time)
            self._print_training_info(training_info)

    def feed(self, inputs, decimal=0, absolute=True):
        one_dimentional = False

        z = inputs

        if len(z.shape) == 1:
            one_dimentional = True
            r = [z]
            z = r

        z = np.transpose(z)

        for i in range(len(self._weights)):
            z = np.add(np.dot(self._weights[i], z), self._biases[i])
            z = self._activation_functions[i + 1](z)

        if one_dimentional:
            result = np.round(np.transpose(z), decimals=decimal)[0]
        else:
            result = np.round(np.transpose(z), decimals=decimal)

        if absolute:
            return np.absolute(result)
        else:
            return result

    def accuracy(self, query="validation", decimal=0):

        query = Data.query(query)

        if query == "training":
            if self._training_data:
                results = np.round(self._feed_forward(self._training_data["features"]), decimals=decimal)
                labels = self._training_data["labels"]
                length = self._training_data["length"]
            else:
                raise Exception("Training data not found.")

        elif query == "validation":
            if self._validation_data:
                results = np.round(self._feed_forward(self._validation_data["features"]), decimals=decimal)
                labels = self._validation_data["labels"]
                length = self._validation_data["length"]
            else:
                raise Exception("Validation data not found.")

        elif query == "test":
            if self._test_data:
                results = np.round(self._feed_forward(self._test_data["features"]), decimals=decimal)
                labels = self._test_data["labels"]
                length = self._test_data["length"]
            else:
                raise Exception("Test data not found.")

        else:
            raise \
                Exception("Query '" + query + "' is not available. Available queries are: training, validation, test.")

        results = np.transpose(results)
        labels = np.transpose(labels)

        accuracy = np.sum(np.all(results == labels, axis=1))

        return int(np.round(accuracy/length*100))

    def cost(self, query="training", decimal=10):

        query = Data.query(query)

        if query == "training":
            if self._training_data:
                features = self._training_data["features"]
                labels = self._training_data["labels"]
                length = self._training_data["length"]
            else:
                raise Exception("Training data not found.")

        elif query == "validation":
            if self._validation_data:
                features = self._validation_data["features"]
                labels = self._validation_data["labels"]
                length = self._validation_data["length"]
            else:
                raise Exception("Validation data not found.")
        elif query == "test":
            if self._test_data:
                features = self._test_data["features"]
                labels = self._test_data["labels"]
                length = self._test_data["length"]
            else:
                raise Exception("Test data not found.")
        else:
            raise \
                Exception("Query '" + query + "' is not available. Available queries are: training, validation, test.")

        cost = self._loss_function(self._feed_forward(features), labels)
        return np.round(np.sum(cost) / length, decimal)

    def _add_training_info(self, training_info, epoch, time):
        training_cost = "NA"
        validation_cost = "NA"
        test_cost = "NA"

        training_accuracy = "NA"
        validation_accuracy = "NA"
        test_accuracy = "NA"

        if "training_cost" in training_info:
            training_cost = self.cost("training")
        if "validation_cost" in training_info:
            validation_cost = self.cost("validation")
        if "test_cost" in training_info:
            test_cost = self.cost("test")
        if "training_accuracy" in training_info:
            training_accuracy = self.accuracy("training")
        if "validation_accuracy" in training_info:
            validation_accuracy = self.accuracy("validation")
        if "test_accuracy" in training_info:
            test_accuracy = self.accuracy("test")

        self._training_info += [{"epoch": epoch,
                                 "time": time,

                                 "training_cost": training_cost,
                                 "validation_cost": validation_cost,
                                 "test_cost": test_cost,

                                 "training_accuracy": training_accuracy,
                                 "validation_accuracy": validation_accuracy,
                                 "test_accuracy": test_accuracy
                                 }]

    def _print_training_info(self, training_info):
        info = ""
        for t in training_info:
            if t in self._training_info[-1].keys():
                info += Data.query(t, True) + ": " + str(self._training_info[-1][t]) + ", "
            else:
                raise Exception("'" + Data.query(t, True) + "' is not an available info. "
                                "All available infos are: " +
                                str(Data.query(list(self._training_info[-1].keys()), True)))
        info = info[:-2]
        print(info)
