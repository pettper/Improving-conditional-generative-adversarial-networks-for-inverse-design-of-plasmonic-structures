import argparse
from pathlib import Path

import torch
from torch.nn import Softplus
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from src.cnn_regression import (
    EfficientNetRegressionV3,
    EfficientNetRegressionV3General,
    EfficientNetV2Regression,
    EfficientNetV2RegressionGeneral,
)
from src.gan import WGANTrainer
from src.utils import DimerDataset as Dataset
from src.utils import (
    DimerVariable,
    count_parameters,
    get_image_size,
    get_label_size,
    initialize_dcgan_weights,
    load_yaml,
)

torch.manual_seed(23)

# The image size 128x128 is constant, adding this line optimizes specific algorithms -> improves training runtime.
torch.backends.cudnn.benchmark = True


class WGANGPMain:
    NUM_WORKERS = 4  # Parallelism for data loaders
    EXPECTED_KEYS = [
        "architecture",
        "target_variable",
        "epochs",
        "batch_size",
        "learning_rate",
        "beta1",
        "beta2",
        "dropout_rate",
        "feature_scaling",
        "save_model_filename",
        "load_model_filename",
        "feed_forward_network",
        "data_dir",
        "tensorboard",
        "use_cbn",
        "use_label_projection",
        "use_embedding_network",
    ]

    def __init__(self, settings: dict):
        # All relevant settings are passed to main through the dictionary 'settings'
        missing = [key for key in self.EXPECTED_KEYS if key not in settings]
        if missing:
            raise KeyError(f"Missing required keys: {', '.join(missing)}")
        else:
            self.settings = settings

        # Decide on which device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.init()

        self.train_loader, self.val_loader = self._setup_data_loaders()
        self.critic, self.generator = self._import_gan()
        self.critic_opt, self.generator_opt = self._setup_optimizers()
        self.forward_network = self._load_forward_network()
        self.print_model_info()

    def run(self, benchmark=False):
        # Create a WGAN-trainer object and run 'train_model' to optimize
        gan_trainer = WGANTrainer(
            self.critic,
            self.generator,
            self.critic_opt,
            self.generator_opt,
            self.train_loader,
            self.val_loader,
            self.device,
            load_model_filename=self.settings["load_model_filename"],
            save_model_filename=self.settings["save_model_filename"],
            forward_network=self.forward_network,
            write_to_tensorboard=self.settings["tensorboard"],
        )

        # Train WGAN-GP model
        epochs = self.settings["epochs"]
        epoch_times = gan_trainer.train_model(epochs, benchmark=benchmark)

        if benchmark:
            print("--- WGAN-GP RUN TIME TEST ---")
            print("Epochs                    = %d" % epochs)
            print("Average runtime per epoch = %.7f" % (torch.mean(epoch_times).item()))
            print(
                "Median runtime per epoch  = %.7f" % (torch.median(epoch_times).item())
            )
            print(f"Epoch times: {epoch_times}")

    def _setup_data_loaders(self):
        # "data/dimer_cylinder_train_val_test/" for example. Expects to find training, validation, and test directories
        root_dir = Path(self.settings["data_dir"])
        train_dir = root_dir.joinpath(Path("training/featherfiles/"))
        val_dir = root_dir.joinpath(Path("validation/featherfiles/"))

        variable = DimerVariable.keymap(self.settings["target_variable"])
        training_dataset = Dataset(train_dir, variable)
        validation_dataset = Dataset(
            val_dir,  # Validation set uses same transforms as in the training set
            variable,
            transform=lambda x: training_dataset.apply_transform(x),
            target_transform=lambda x: training_dataset.apply_target_transform(x),
        )

        batch_size = self.settings["batch_size"]
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(training_dataset),
            pin_memory=True,
            num_workers=self.NUM_WORKERS,
            persistent_workers=True,
            drop_last=True
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(validation_dataset),
            pin_memory=True,
            num_workers=self.NUM_WORKERS,
            persistent_workers=True,
        )

        return training_loader, validation_loader

    def _import_gan(self):
        # Handle imports
        model_type = self.settings["architecture"]
        if model_type == "dc":
            from src.gan.dcgan import DCGANCritic as Critic
            from src.gan.dcgan import DCGANGenerator as Generator
        elif model_type == "fc":
            from src.gan.fcgan import FullyConnectedCritic as Critic
            from src.gan.fcgan import FullyConnectedGenerator as Generator
        elif model_type == "fcskip":
            from src.gan.fcgan import FullyConnectedCritic as Critic
            from src.gan.fcgan import FullyConnectedGeneratorSkip as Generator
        elif model_type == "eff":
            from src.gan.effnetgan import EffNetCritic as Critic
            from src.gan.effnetgan import EffNetGenerator as Generator
        elif model_type == "effv2":
            from src.gan.effnetgan import EffNetV2Critic as Critic
            from src.gan.effnetgan import EffNetV2Generator as Generator
        else:
            raise Exception("No model architecture was specified.")

        feature_scaling = self.settings["feature_scaling"]
        dropout_rate = self.settings["dropout_rate"]
        channels, image_size, _ = get_image_size(self.train_loader)
        target_channels, ydim = get_label_size(self.train_loader)

        # Initialize critic
        critic = Critic(
            channels,
            target_channels=target_channels,
            image_size=image_size,
            y_dim=ydim,
            features=feature_scaling,
            lp=self.settings["use_label_projection"],
            embed=self.settings["use_embedding_network"],
            dropout_rate=dropout_rate,
        ).to(self.device)

        # Initialize generator
        if model_type == "dc":
            generator = Generator(
                channels,
                target_channels=target_channels,
                image_size=image_size,
                y_dim=ydim,
                features=feature_scaling,
                use_cbn=self.settings["use_cbn"],
                dropout_rate=dropout_rate,
            ).to(self.device)
        else:
            generator = Generator(
                channels,
                target_channels=target_channels,
                image_size=image_size,
                y_dim=ydim,
                features=feature_scaling,
                dropout_rate=dropout_rate,
            ).to(self.device)

        # Initialize weights
        initialize_dcgan_weights(critic)
        initialize_dcgan_weights(generator)

        return critic, generator

    def _setup_optimizers(self):
        learning_rate = self.settings["learning_rate"]
        b1 = self.settings["beta1"]
        b2 = self.settings["beta2"]

        # Setup an Adam-optimizer for the critic and the generator
        critic_optim = Adam(self.critic.parameters(), lr=learning_rate, betas=(b1, b2))
        generator_optim = Adam(
            self.generator.parameters(), lr=learning_rate, betas=(b1, b2)
        )

        return critic_optim, generator_optim

    def _load_forward_network(self):
        """
        Load an optional feed forward neural network to use a metric when training the conditional GAN models."
        Returns 'None' if no such network is provided. Expects self.settings["feed_forward_network"] to be a
        string for a path to a saved model in a pth.tar-file.
        """
        target_channels, ydim = get_label_size(self.train_loader)
        target_variable = DimerVariable.keymap(self.settings["target_variable"])
        im_channels, image_size, _ = get_image_size(self.train_loader)

        # Optional feed forward network to use as evaluation metric
        forward_network = None
        if self.settings["feed_forward_network"] is not None:
            checkpoint = torch.load(
                self.settings["feed_forward_network"],
                map_location=self.device,
                weights_only=False,
            )
            try:
                if (
                    target_variable == DimerVariable.ALL
                    or target_variable == DimerVariable.CROSS_SECTIONS
                ):
                    forward_network = EfficientNetRegressionV3General(
                        im_channels,
                        ydim,
                        activation=Softplus(),
                        image_size=image_size,
                        out_channels=target_channels,
                        dropout_rate=0.5,
                    ).to(self.device)
                    forward_network.load_state_dict(checkpoint["model_state_dict"])
                    print(
                        f"Succesfully loaded forward network as EfficientNet for {target_variable.name} dimer variables!"
                    )
                else:
                    forward_network = EfficientNetRegressionV3(
                        im_channels,
                        ydim,
                        activation=Softplus(),
                        image_size=image_size,
                        dropout_rate=0.5,
                    ).to(self.device)
                    forward_network.load_state_dict(checkpoint["model_state_dict"])
                    print("Succesfully loaded forward network as EfficientNet!")
            except Exception:
                print(
                    "Could not load given forward network as EfficientNet, tries to load as EfficientNetV2..."
                )
                try:
                    if (
                        target_variable == DimerVariable.ALL
                        or target_variable == DimerVariable.CROSS_SECTIONS
                    ):
                        forward_network = EfficientNetV2RegressionGeneral(
                            im_channels,
                            ydim,
                            activation=Softplus(),
                            image_size=image_size,
                            out_channels=target_channels,
                            dropout_rate=0.5,
                        ).to(self.device)
                        forward_network.load_state_dict(checkpoint["model_state_dict"])
                        print(
                            f"...Succesfully loaded forward network as EfficientNetV2 for {target_variable.name} dimer variables!"
                        )
                    else:
                        forward_network = EfficientNetV2Regression(
                            im_channels,
                            ydim,
                            activation=Softplus(),
                            image_size=image_size,
                            dropout_rate=0.5,
                        ).to(self.device)
                        forward_network.load_state_dict(checkpoint["model_state_dict"])
                        print(
                            "...Succesfully loaded forward network as EfficientNetV2!"
                        )
                except Exception:
                    raise Exception(
                        "The feed forward network is not of type EffNetV2 or EffNet. Can also be that the target variable is not supported."
                    )
        return forward_network

    def print_model_info(self):
        print(f"Device: {self.device}")
        print(f"Architecture: {self.settings['architecture']}")

        print(f"Label projection: {self.settings['use_label_projection']}")
        print(f"Embedding network: {self.settings['use_embedding_network']}")
        print(f"Conditional batch normalization: {self.settings['use_cbn']}")

        # Print parameters
        critic_params = count_parameters(self.critic)
        gen_params = count_parameters(self.generator)
        print(f"Generator parameters: {gen_params}")
        print(f"Critic parameters: {critic_params}")


if __name__ == "__main__":
    # Configure command line arguments
    parser = argparse.ArgumentParser(
        prog="wgan_gp_main.py",
        description="The program trains a Wasserstein GAN model on user"
        + " specified data for a given number of epochs.",
    )
    parser.add_argument(
        "-s",
        "--settings",
        type=str,
        default="./settings/wgan_gp_example.yaml",
        help="Path to a settings.yaml file.",
    )
    args = parser.parse_args()

    settings = load_yaml(args.settings)
    main = WGANGPMain(settings)

    main.run()  # Main entry point
