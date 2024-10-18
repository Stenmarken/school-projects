import unittest
from N_dimensional_network_remade import N_dimensional_network
from lab3.lab3 import read_parameters, read_data
import numpy as np
class TestNDimensionalNetworkMethods(unittest.TestCase):

    def test_create_matrices(self):
        parameters = read_parameters("test_yaml.yaml")
        train_data, val_data, test_data, data_batches = read_data(parameters)
        n_dim = N_dimensional_network(parameters=parameters, 
                                          data_batches=data_batches,
                                          train_data=train_data, 
                                          val_data=val_data,
                                          test_data=test_data)
        assert n_dim.Ws[0].shape == (50, 3072)
        assert n_dim.Ws[1].shape == (30, 50)
        assert n_dim.Ws[2].shape == (20, 30), f"n_dim.Ws[2].shape: {n_dim.Ws[2].shape}"
        assert n_dim.Ws[3].shape == (20, 20)
        assert n_dim.Ws[4].shape == (10, 20)
        assert n_dim.Ws[5].shape == (10, 10)
        assert n_dim.Ws[6].shape == (10, 10)
        assert n_dim.Ws[7].shape == (10, 10)
        assert n_dim.Ws[8].shape == (10, 10)


if __name__ == '__main__':
    unittest.main()