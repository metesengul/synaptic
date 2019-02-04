import numpy as np


class Data:

    @staticmethod
    def save(data, name):
        np.save(name, data)

    @staticmethod
    def load(name, validation_percentage=0, test_percentage=0):
        data = np.load(name + ".npy")
        if validation_percentage != 0 or test_percentage != 0:
            data = Data.slice(data, validation_percentage, test_percentage)
            return data
        else:
            return data.item()

    @staticmethod
    def query(query, reverse=False):
        if type(query) == list or type(query) == tuple:
            new_query = []
            for q in query:
                new_query += [Data.query(q, reverse)]
        elif type(query) == dict:
            new_query = {}
            for k in query.keys():
                new_query[Data.query(k, reverse)] = query[k]
        else:
            if reverse:
                new_query = query.lower()
                new_query = new_query.replace("_", " ")
                new_query = new_query.replace("-", " ")
                new_query = new_query.title()
            else:
                new_query = query.lower()
                new_query = new_query.replace(" ", "_")
                new_query = new_query.replace("-", "_")
        return new_query

    @staticmethod
    def to_onehot(x, classes):
        nb_classes = classes
        targets = x.reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets]
        return one_hot_targets

    @staticmethod
    def shuffle(data):
        random_state = np.random.get_state()
        np.random.shuffle(data["features"])
        np.random.set_state(random_state)
        np.random.shuffle(data["labels"])

    @staticmethod
    def slice(data, validation_percentage, test_percentage):
        data = data.item()
        data_length = len(data["features"])
        validation_count = int(validation_percentage / 100 * data_length)
        test_count = int(test_percentage / 100 * data_length)

        validation_features = data["features"][:validation_count]
        test_features = data["features"][validation_count:validation_count + test_count]
        training_features = data["features"][validation_count + test_count:]

        validation_labels = data["labels"][:validation_count]
        test_labels = data["labels"][validation_count:validation_count + test_count]
        training_labels = data["labels"][validation_count + test_count:]

        validation_set = {"features": validation_features, "labels": validation_labels}
        test_set = {"features": test_features, "labels": test_labels}
        training_set = {"features": training_features, "labels": training_labels}

        return {"validation": validation_set, "test": test_set, "training": training_set}

    @staticmethod
    def normalize(x, u="NA", s="NA"):
        if u == "NA" or s == "NA":
            u = np.mean(x)
            s = np.mean((x - u) ** 2)
        r = (x - u) / s
        return r


class ActivationFunctions:

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return ActivationFunctions.sigmoid(x) * (1 - ActivationFunctions.sigmoid(x))

        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - (ActivationFunctions.tanh(x) ** 2)

        return (np.e ** np.multiply(x, 2) - 1) / (np.e ** np.multiply(x, 2) + 1)

    @staticmethod
    def softmax(x, derivative=False):
        if derivative:
            return ActivationFunctions.softmax(x) * (1 - ActivationFunctions.softmax(x))

        return np.exp(x) / (np.ones(shape=np.shape(x)) * np.sum(np.exp(x)))

    @staticmethod
    def linear(x, derivative=False):
        if derivative:
            return np.ones(shape=(np.shape(x)))

        return x

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.greater(x, 0).astype(int)

        return np.maximum(x, 0)


class LossFunctions:

    @staticmethod
    def logistic(a, y, derivative=False):
        if derivative:
            return (np.subtract(1, y) / np.subtract(1, a)) - np.divide(y, a)
        return - (np.multiply(y, np.log(a)) + np.multiply(np.subtract(1, y), np.log(np.subtract(1, a))))

    @staticmethod
    def mean_squared(a, y, derivative=False):
        if derivative:
            return np.subtract(a, y)
        return (np.subtract(a, y) ** 2) / 2


class InitializationFunctions:

    @staticmethod
    def relu_optimized(index, layers):
        return np.sqrt(2 / layers[index])

    @staticmethod
    def tanh_optimized(index, layers):
        return np.sqrt(1 / layers[index])

    @staticmethod
    def xavier(index, layers):
        return np.sqrt(2 / (layers[index] + layers[index + 1]))


class Optimizers:

    @staticmethod
    def _default_parameters(default, given):
        result = default
        for k in default.keys():
            if k in given:
                result[k] = given[k]

        for k in given.keys():
            if k not in default.keys():
                raise Exception(
                    "Key '" + Data.query(k, True) + "' is not used as a parameter. "
                    "All parameters and their defalut values are: " + str(Data.query(default, True)))

        return result

    @staticmethod
    def _decay(learning_rate, model, parameters):
        new_learning_rate = learning_rate / (1 + parameters["decay"] * (model.training_info[-1]["epoch"] + 1))
        return new_learning_rate

    class Standart:
        space = 0
        default = {"learning_rate": 0.001, "decay": 0}

        @staticmethod
        def run(parameters, model):

            default = Optimizers.Momentum.default
            parameters = Optimizers._default_parameters(default, parameters)

            learning_rate = Optimizers._decay(parameters["learning_rate"], model, parameters)

            weights, biasas = model.weights, model.biases
            dw, db = model.dw, model.db
            length = len(weights)

            fdw = dw[0]
            fdb = db[0]

            for i in range(length):
                weights[i] = weights[i] - fdw[i] * learning_rate
                biasas[i] = biasas[i] - fdb[i] * learning_rate

    class Momentum:
        space = 1
        default = {"momentum": 0.9, "learning_rate": 0.001, "decay": 0}

        @staticmethod
        def run(parameters, model):

            default = Optimizers.Momentum.default
            parameters = Optimizers._default_parameters(default, parameters)

            momentum = parameters["momentum"]
            learning_rate = Optimizers._decay(parameters["learning_rate"], model, parameters)

            weights, biasas = model.weights, model.biases
            dw, db = model.dw, model.db
            length = len(weights)

            fdw = dw[0]
            vdw = dw[1]

            fdb = db[0]
            vdb = db[1]

            for i in range(length):
                vdw[i] = momentum * vdw[i] + (1 - momentum) * fdw[i]
                vdb[i] = momentum * vdb[i] + (1 - momentum) * fdb[i]

                weights[i] = weights[i] - vdw[i] * learning_rate
                biasas[i] = biasas[i] - vdb[i] * learning_rate

    class RMSProp:
        space = 1
        default = {"momentum": 0.9, "learning_rate": 0.001, "decay": 0}

        @staticmethod
        def run(parameters, model):

            default = Optimizers.RMSProp.default
            parameters = Optimizers._default_parameters(default, parameters)

            momentum = parameters["momentum"]
            learning_rate = Optimizers._decay(parameters["learning_rate"], model, parameters)

            weights, biasas = model.weights, model.biases
            dw, db = model.dw, model.db
            length = len(weights)

            fdw = dw[0]
            sdw = dw[1]

            fdb = db[0]
            sdb = db[1]

            for i in range(length):
                sdw[i] = momentum * sdw[i] + (1 - momentum) * np.square(fdw[i])
                sdb[i] = momentum * sdb[i] + (1 - momentum) * np.square(fdb[i])

                weights[i] = weights[i] - fdw[i] / (np.sqrt(sdw[i]) + 1e-8) * learning_rate
                biasas[i] = biasas[i] - fdb[i] / (np.sqrt(sdb[i]) + 1e-8) * learning_rate

    class Adam:
        space = 2
        default = {"b1": 0.9, "b2": 0.9, "learning_rate": 0.001, "decay": 0}

        @staticmethod
        def run(parameters, model):

            default = Optimizers.Adam.default
            parameters = Optimizers._default_parameters(default, parameters)

            batch_size = model.batch_size

            learning_rate = Optimizers._decay(parameters["learning_rate"], model, parameters)

            b1 = parameters["b1"]
            b2 = parameters["b2"]

            weights, biasas = model.weights, model.biases
            dw, db = model.dw, model.db
            length = len(weights)

            fdw = dw[0]
            vdw = dw[1]
            sdw = dw[2]

            fdb = db[0]
            vdb = db[1]
            sdb = db[2]

            for i in range(length):
                vdw[i] = b1 * vdw[i] + (1 - b1) * fdw[i]
                vdb[i] = b1 * vdb[i] + (1 - b1) * fdb[i]

                sdw[i] = b2 * sdw[i] + (1 - b2) * np.square(fdw[i])
                sdb[i] = b2 * sdb[i] + (1 - b2) * np.square(fdb[i])

                vdw[i] = vdw[i] / (1 - b1 ** batch_size)
                vdb[i] = vdb[i] / (1 - b1 ** batch_size)

                sdw[i] = sdw[i] / (1 - b2 ** batch_size)
                sdb[i] = sdb[i] / (1 - b2 ** batch_size)

                weights[i] = weights[i] - vdw[i] / (np.sqrt(sdw[i]) + 1e-8) * learning_rate
                biasas[i] = biasas[i] - vdb[i] / (np.sqrt(sdb[i]) + 1e-8) * learning_rate
