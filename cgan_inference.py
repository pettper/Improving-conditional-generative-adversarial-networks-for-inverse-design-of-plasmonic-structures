from pathlib import Path

import torch
from torch.nn import Softplus
from torch.utils.data import DataLoader, RandomSampler

from src.cnn_regression import EfficientNetV2RegressionGeneral
from src.gan.dcgan.dcgan_generator import DCGANGenerator
from src.gan.fcgan.fc_generator import FullyConnectedGenerator as FCGANGenerator
from src.utils import DimerDataset as Dataset
from src.utils import (
    DimerVariable,
    estimate_forward_error,
    estimate_reconstruction_error,
    get_image_size,
    get_label_size,
)


class CGANInference:
    def __init__(
        self,
        fcgan_checkpoints_dict,
        dcgan_checkpoints_dict,
        forward_network,
        train_dataset,
        val_dataset,
        test_dataset,
        fcgan_feature_scaling=1,
        dcgan_feature_scaling=1,
        device="cpu",
    ):
        """
        Expects a dictionary of path strings to stored model checkpoints.
        INPUTS:
            fcgan_checkpoints_dict: A dictionary of 2-tuples, (checkpoint_filename, dropout). The keys will be treated as labels.
            dcgan_checkpoints_dict: A dictionary of 2-tuples, (checkpoint_filename, dropout). The keys will be treated as labels.
            train_dataset: Training dataset, an instance of DimerDataset.
            val_dataset: Validation dataset, an instance of DimerDataset.
            device: Torch device to use, defaults to "cpu".
        """
        self.device = device

        # Load checkpoints and dropout rates
        self.fcgan_checkpoints = self._load_checkpoints(fcgan_checkpoints_dict)
        self.dcgan_checkpoints = self._load_checkpoints(dcgan_checkpoints_dict)

        # Dataset and data loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = CGANInference._setup_data_loader(
            self.train_dataset, RandomSampler(self.train_dataset)
        )
        self.val_loader = CGANInference._setup_data_loader(
            self.val_dataset, RandomSampler(self.val_dataset)
        )
        self.test_loader = CGANInference._setup_data_loader(self.test_dataset)

        # Forward network is used for evaluation
        self.forward_network = self._load_forward_network(forward_network)
        self.forward_network.eval()

        # Feature scaling
        self.fc_features = fcgan_feature_scaling
        self.dc_features = dcgan_feature_scaling

    def calculate_metrics(self):
        result = {}
        with torch.no_grad():
            for cp, t in zip(
                [self.fcgan_checkpoints, self.dcgan_checkpoints], ["fc", "dc"]
            ):
                for k, state in cp.items():
                    g = self._load_generator(
                        state["generator_state_dict"],
                        type=t,
                        dropout_rate=state["dropout"],
                    )
                    g.eval()
                    tmp_dict = {}
                    for data_key, data_loader in zip(
                        ["training", "validation", "test"],
                        [self.train_loader, self.val_loader, self.test_loader],
                    ):
                        im_error = estimate_reconstruction_error(
                            g, data_loader, self.device, n=3
                        )
                        fw_error = estimate_forward_error(
                            g, data_loader, self.device, self.forward_network, n=3
                        )
                        tmp_dict[data_key] = {
                            "forward_error": fw_error[0],
                            "image_error": im_error[0],
                            "masked_image_error": im_error[2],
                        }
                    result[k] = tmp_dict
        return result

    def _load_generator(self, state_dict, type="dc", dropout_rate=0.5):
        im_ch, im_size, _ = get_image_size(self.train_loader)
        lab_ch, ydim = get_label_size(self.train_loader)
        if type == "dc":
            generator = DCGANGenerator(
                im_ch,
                target_channels=lab_ch,
                y_dim=ydim,
                image_size=im_size,
                dropout_rate=dropout_rate,
                features=self.dc_features,
            )
        elif type == "fc":
            generator = FCGANGenerator(
                im_ch,
                target_channels=lab_ch,
                y_dim=ydim,
                image_size=im_size,
                dropout_rate=dropout_rate,
                features=self.fc_features,
            )
        else:
            raise TypeError(f"Unknown model type: {type}")
        generator.load_state_dict(state_dict)
        return generator.to(self.device)

    def _load_forward_network(self, forward_network_dict):
        im_ch, im_size, _ = get_image_size(self.train_loader)
        out_ch, ydim = get_label_size(self.train_loader)

        fn_checkpoint = torch.load(
            forward_network_dict["load_path"],
            weights_only=False,
        )
        fn = forward_network_dict["model"]
        forward_network = fn(
            im_ch,
            ydim,
            activation=Softplus(),
            image_size=im_size,
            out_channels=out_ch,
            dropout_rate=0.5,
        )
        forward_network.load_state_dict(fn_checkpoint["model_state_dict"])
        return forward_network.to(self.device)

    def _load_checkpoints(self, checkpoints_dict):
        checkpoints = {}
        for k in checkpoints_dict.keys():
            checkpoints[k] = torch.load(
                checkpoints_dict[k][0], map_location=self.device, weights_only=False
            )
            checkpoints[k]["dropout"] = checkpoints_dict[k][1]
        return checkpoints

    @staticmethod
    def _setup_data_loader(dataset, sampler=None):
        loader = None
        if dataset:
            loader = DataLoader(
                dataset,
                batch_size=3000,
                sampler=sampler,
                pin_memory=True,
            )
        return loader


if __name__ == "__main__":
    root_dir = Path("./data/anisotropic_au_structures_train_val_test")
    train_dir = root_dir.joinpath(Path("training/featherfiles/"))
    val_dir = root_dir.joinpath(Path("validation/featherfiles/"))
    test_dir = root_dir.joinpath(Path("test/featherfiles/"))

    train_dataset = Dataset(train_dir, DimerVariable.CROSS_SECTIONS)
    val_dataset = Dataset(
        val_dir,  # Validation set uses same transforms as in the training set
        DimerVariable.CROSS_SECTIONS,
        transform=lambda x: train_dataset.apply_transform(x),
        target_transform=lambda x: train_dataset.apply_target_transform(x),
        inverse_transform=lambda x: train_dataset.apply_inverse_transform(x),
        inverse_target_transform=lambda x: train_dataset.apply_inverse_target_transform(
            x
        ),
    )
    test_dataset = Dataset(
        test_dir,  # Validation set uses same transforms as in the training set
        DimerVariable.CROSS_SECTIONS,
        transform=lambda x: train_dataset.apply_transform(x),
        target_transform=lambda x: train_dataset.apply_target_transform(x),
        inverse_transform=lambda x: train_dataset.apply_inverse_transform(x),
        inverse_target_transform=lambda x: train_dataset.apply_inverse_target_transform(
            x
        ),
    )

    forward_network = {
        "load_path": "delivery/pretrained_cnn_models/last_epoch_effv2_cross_all_structures_lr00001_drop05.pth.tar",
        "model": EfficientNetV2RegressionGeneral,
    }

    fcgan_files = {
        "FCGAN": (
            "delivery/aip_review_results/all_structures/fcgan_lp=0_em=0_data=all_drop=0.0_feat=4_last_epoch_20000.pth.tar",
            0.0,
        ),
        "FCGAN + LP": (
            "delivery/aip_review_results/all_structures/fcgan_lp=1_em=0_data=all_drop=0.0_feat=4_last_epoch_20000.pth.tar",
            0.0,
        ),
        "FCGAN + Em.": (
            "delivery/aip_review_results/all_structures/fcgan_lp=0_em=1_data=all_drop=0.0_feat=4_last_epoch_20000.pth.tar",
            0.0,
        ),
        "FCGAN + LP + Em.": (
            "delivery/aip_review_results/all_structures/fcgan_lp=1_em=1_data=all_drop=0.0_feat=4_last_epoch_20000.pth.tar",
            0.0,
        ),
    }

    dcgan_files = {
        "DCGAN": (
            "delivery/aip_review_results/all_structures/dcgan_lp=0_em=0_data=all_drop=0.0_last_epoch_20000.pth.tar",
            0.0,
        ),
        "DCGAN + LP": (
            "delivery/aip_review_results/all_structures/dcgan_lp=1_em=0_data=all_drop=0.0_last_epoch_20000.pth.tar",
            0.0,
        ),
        "DCGAN + Em.": (
            "delivery/aip_review_results/all_structures/dcgan_lp=0_em=1_data=all_drop=0.0_last_epoch_20000.pth.tar",
            0.0,
        ),
        "DCGAN + LP + Em.": (
            "delivery/aip_review_results/all_structures/dcgan_lp=1_em=1_data=all_drop=0.0_last_epoch_20000.pth.tar",
            0.0,
        ),
    }

    cgan_inference = CGANInference(
        fcgan_files,
        dcgan_files,
        forward_network,
        train_dataset,
        val_dataset,
        test_dataset,
        fcgan_feature_scaling=4,
    )
    result = cgan_inference.calculate_metrics()
    print(result)
