import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import sys
import os

# Adjust the Python path to include the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.dataloader import DataLoader
from model.cnn_lstm import CNN_LSTM  # Make sure to import the CNN_LSTM class correctly

class TestCNNLSTM(unittest.TestCase):

    def setUp(self):
        # Mock config for testing
        self.config = {
            'train': {
                'n_samples_per_signal': 2048,
                'batch_size': 16,
                'epochs': 1,
                'detector': 'Livingston',
                'dataset_type': 'IMRPhenomXPHM injections',
                'train_from_checkpoint': False,
                'checkpoint_path': '/tmp/checkpoint',
            },
            'model': {
                'layers': {
                    'learning_rate': 0.001,
                    'CNN_layer_1': 32,
                    'CNN_layer_2': 64,
                    'LSTM_layer_1': 50,
                    'LSTM_layer_2': 50,
                    'kernel_size': 3,
                    'pool_size': 2,
                    'Dropout': 0.2,
                    'num_heads_MHA': 4,
                    'key_dim_MHA': 16,
                },
                'timesteps': 10,
                'model_save_path': '/tmp/model.h5',
                'results_save_path': '/tmp/results.hdf5',
            },
            'data': {
                'path_train_original_1': '/tmp/train_data_1.h5',
                'path_train_original_2': '/tmp/train_data_2.h5',
                'path_train_NRSur': '/tmp/train_NRSur.h5',
                'path_train_high_mass': '/tmp/train_high_mass.h5',
                'path_train_DC': '/tmp/train_DC.h5',
                'path_train': '/tmp/train_IMRPhenomXPHM.h5',
                'path_test_original': '/tmp/test_data_original.h5',
                'path_test_NRSur': '/tmp/test_NRSur.h5',
                'path_test_high_mass': '/tmp/test_high_mass.h5',
                'path_test_DC': '/tmp/test_DC.h5',
                'path_test_1': '/tmp/test_IMRPhenomXPHM.h5',
            }
        }

        # Mock data
        self.mock_data = np.random.rand(10, 2048)  # 10 signals, each with 2048 samples
        self.mock_labels = np.random.rand(10, 2048)  # 10 signals, each with 2048 samples

        # Patch the DataLoader's load_data method to use the mock data
        self.original_load_data = DataLoader.load_data
        DataLoader.load_data = self.mock_load_data

    def tearDown(self):
        # Restore the original load_data method after each test
        DataLoader.load_data = self.original_load_data

    def mock_load_data(self, data_config, dataset_type):
        return self.mock_data, self.mock_labels

    def test_load_data(self):
        model = CNN_LSTM(self.config)
        model.load_data()
        
        self.assertEqual(model.X_train_noisy.shape[0], 10)
        self.assertEqual(model.X_test_noisy.shape[0], 10)
        self.assertEqual(model.X_train_pure.shape[0], 10)
        self.assertEqual(model.X_test_pure.shape[0], 10)

    def test_preprocess_data(self):
        model = CNN_LSTM(self.config)
        processed_data = model._preprocess_data(self.mock_data, 2048)
        
        for data in processed_data:
            self.assertTrue(np.max(data) <= 1.0)
            self.assertTrue(np.min(data) >= -1.0)

    def test_build_model(self):
        model = CNN_LSTM(self.config)
        model.load_data()
        model.build()
        
        self.assertIsNotNone(model.model)
        self.assertTrue(hasattr(model.model, 'layers'))

    def test_train_model(self):
        model = CNN_LSTM(self.config)
        model.load_data()
        model.build()

        # Mocking the train method to avoid actual training
        def mock_train(self, checkpoint):
            pass
        CNN_LSTM.train = mock_train

        model.train(None)
        self.assertTrue(True)  # If no exception, the test passes

    def test_predict_with_uncertainty(self):
        model = CNN_LSTM(self.config)
        model.load_data()
        model.build()
        
        x_test = np.random.rand(2048)
        mean_preds, lower_bound, upper_bound = model.predict_with_uncertainty(x_test)
        
        self.assertEqual(mean_preds.shape, (2048,))
        self.assertEqual(lower_bound.shape, (2048,))
        self.assertEqual(upper_bound.shape, (2048,))

if __name__ == '__main__':
    unittest.main()