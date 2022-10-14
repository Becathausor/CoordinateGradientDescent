import dataclasses
import numpy as np


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 proportion: float = 0.5, shuffle: bool = True):
        """Create a Dataset
        Parameters:
            X
            """
        self.X: np.array = X
        self.y: np.array = y
        self.data = np.array(list(zip(self.X, self.y)))

        self.proportion: float = proportion
        self.shuffle: bool = shuffle

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.data_train = None
        self.data_test = None

        if shuffle:
            self.mix_all()

        self.separate_data()

    def separate_data(self):
        threshold = len(self.data) * self.proportion
        self.data_train = self.data[:threshold]
        self.data_test = self.data[threshold:]

        # Unzip
        self.X_train, self.y_train = zip(*self.data_train)
        self.X_test, self.y_test = zip(*self.data_test)

    def mix_all(self):
        assert self.data is not None, "Initialisation error"
        self.data = list(np.random.shuffle(self.data))
        self.separate_data()

    def shuffle_train(self):
        assert self.data_train is not None, "Initialisation error"
        self.data_train = list(np.random.shuffle(self.data_train))
        self.actualize_data()

    def shuffle_test(self):
        assert self.data_test is not None, "Initialisation error"
        self.data_test = list(np.random.shuffle(self.data_test))
        self.actualize_data()

    def combine_dataset(self, dataset):
        # TODO: FUSIOOOOOOOONNNNN
        X_train_1 = self.get_X_train()
        X_test_1 = self.get_X_test()

        X_train_2 = dataset.get_X_train()
        X_test_2 = dataset.get_X_test()

        X_train = np.concatenate([X_train_1, X_train_2])
        X_test = np.concatenate([X_test_1, X_test_2])

        raise NotImplementedError
        
    def actualize_data(self, modifier):
        def wrapped(*args):
            result = modifier(args)
            n_train = len(self.X_train)
            n_test = len(self.X_test)
            self.data = [None] * (n_train + n_test)
            self.data[:n_train], self.data[n_train:] = self.data_train, self.data_test
            self.separate_data()
            return result

        return wrapped

    # Getters
    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    #Setters
    @actualize_data
    def set_data_train(self, data: list):
        self.data_train = data

    @actualize_data
    def set_data_test(self, data: list):
        self.data_test = data

    @actualize_data
    def set_data_test(self, data: list):
        self.data = data


if __name__ == '__main__':
    X = np.random.multivariate_normal(np.array([-1, 0]), np.eye(2), 10), \
        np.random.multivariate_normal(np.array([0, 1]), np.eye(2), 40)
    y = np.array([0] * 10 + [1] * 40)
    dataset = Dataset(X, y)
