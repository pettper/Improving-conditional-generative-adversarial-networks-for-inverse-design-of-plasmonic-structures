import unittest
import torch
from src.utils import CustomDataset, CustomDatasetV2, DimerDataset, DimerVariable
from pathlib import Path

class TestDataset(unittest.TestCase):

    def test_dimer_dataset_training_set_transform(self):
        root_dir = Path("data/dimer_cylinder_train_val_test/")
        train_dir = root_dir.joinpath(Path("training/featherfiles/"))

        variable = DimerVariable.CROSS_SECTIONS
        training_dataset = DimerDataset(train_dir, variable)

        # Image data, training set
        self.assertTrue(training_dataset.image_data[:, 0].max() == 1.0)
        self.assertTrue(training_dataset.image_data[:, 0].min() == -1.0)
        self.assertTrue(training_dataset.image_data[:, 1].max() == 1.0)
        self.assertTrue(training_dataset.image_data[:, 1].min() == -1.0)
        # Target data, training set
        self.assertTrue(training_dataset.output_data[:, 0].max() == 1.0)
        self.assertTrue(training_dataset.output_data[:, 0].min() == 0.0)
        self.assertTrue(training_dataset.output_data[:, 1].max() == 1.0)
        self.assertTrue(training_dataset.output_data[:, 1].min() == 0.0)

    def test_dimer_dataset_validation_set_transform(self):
        root_dir = Path("data/dimer_cylinder_train_val_test/")
        train_dir = root_dir.joinpath(Path("training/featherfiles/"))
        val_dir = root_dir.joinpath(Path("validation/featherfiles/"))

        variable = DimerVariable.CROSS_SECTIONS
        training_dataset = DimerDataset(train_dir, variable)
        validation_dataset = DimerDataset(val_dir, variable,
                                          transform=lambda x: training_dataset.apply_transform(x),
                                          target_transform=lambda x: training_dataset.apply_target_transform(x))

        # Check image transforms
        an_image_tensor = torch.randint(low=-10, high=10, size=(1, 2, 128, 128))
        another_image_tensor = torch.randn(size=(1, 2, 128, 128))
        self.assertTrue(torch.equal(
            training_dataset.apply_transform(an_image_tensor),
            validation_dataset.apply_transform(an_image_tensor)))
        self.assertTrue(torch.equal(
            training_dataset.apply_transform(another_image_tensor),
            validation_dataset.apply_transform(another_image_tensor)
        ))
        # Check target transform
        a_target_tensor = torch.randint(low=-10, high=10, size=(1, 2, 81))
        another_target_tensor = torch.randn(size=(1, 2, 81))
        self.assertTrue(torch.equal(
            training_dataset.apply_target_transform(a_target_tensor),
            validation_dataset.apply_target_transform(a_target_tensor)
        ))
        self.assertTrue(torch.equal(
            training_dataset.apply_target_transform(another_target_tensor),
            validation_dataset.apply_target_transform(another_target_tensor)
        ))


if __name__ == '__main__':
    unittest.main()
