from torch.nn import Softplus
from src.cnn_regression import (EfficientNetRegressionV3, EfficientNetV2Regression, EfficientNetRegressionV3General,
                                EfficientNetV2RegressionGeneral)
from src.gan import WGANTrainer
from src.utils import count_parameters, DimerDataset as Dataset, DimerVariable, initialize_dcgan_weights
import torch
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam
import argparse
from pathlib import Path
import yaml

# Function to load settings from yaml-file
def load_yaml(settings_path):
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {settings_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {settings_path}: {e}")

# Configure command line arguments
parser = argparse.ArgumentParser(prog='wgan_gp_main.py', description="The program trains a Wasserstein GAN model on user"
                                                                     + " specified data for a given number of epochs.")
parser.add_argument("-s", "--settings", type=str, default="./settings/wgan_gp_example.yaml", help="Path to a settings.yaml file.")
args = parser.parse_args()

settings = load_yaml(args.settings)

# Import desired model
model_type = settings["architecture"]
if model_type == "dc":
    from src.gan.dcgan import DCGANGenerator as Generator, DCGANCritic as Critic
elif model_type == "fc":
    from src.gan.fcgan import FullyConnectedGenerator as Generator, FullyConnectedCritic as Critic
elif model_type == "fcskip":
    from src.gan.fcgan import FullyConnectedGeneratorSkip as Generator, FullyConnectedCritic as Critic
elif model_type == "eff":
    from src.gan.effnetgan import EffNetGenerator as Generator, EffNetCritic as Critic
elif model_type == "effv2":
    from src.gan.effnetgan import EffNetV2Generator as Generator, EffNetV2Critic as Critic
else:
    raise Exception("No model architecture was specified.")

# Set seed
torch.manual_seed(23)

# The image size 128x128 is constant, adding this line optimizes specific algorithms -> improves training runtime.
torch.backends.cudnn.benchmark = True
NUM_WORKERS=4

# Parameters
CHANNELS = 2
Y_DIM = 81
IMAGE_SIZE = 128
Z_DIM = 100
FEATURES=settings["feature_scaling"]

# Hyperparameters according to WGAN-paper
LEARNING_RATE = settings["learning_rate"]
B1 = settings["beta1"]
B2 = settings["beta2"]
BATCH_SIZE = settings["batch_size"]
EPOCHS = settings["epochs"]
DROPOUT_RATE = settings["dropout_rate"]

# Setup dataloader
# "data/dimer_cylinder_train_val_test/" for example. Expects to find training, validation, and test directories
variable = DimerVariable.keymap(settings["variable"])
root_dir = Path(settings["data_dir"])
train_dir = root_dir.joinpath(Path("training/featherfiles/"))
val_dir = root_dir.joinpath(Path("validation/featherfiles/"))

training_dataset = Dataset(train_dir, variable)
validation_dataset = Dataset(val_dir,  # Validation set uses same transforms as in the training set
                             variable,
                             transform=lambda x: training_dataset.apply_transform(x),
                             target_transform=lambda x: training_dataset.apply_target_transform(x))

training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(training_dataset),
                             pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(validation_dataset),
                               pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=True)

# Model
if settings["use_label_projection"] == "True":
    use_lp = True
else:
    use_lp = False
if settings["use_cbn"] == "True":
    use_cbn = True
else:
    use_cbn = False
if settings["use_embedding_network"] == "True":
    use_embedding_net = True
else:
    use_embedding_net = False

# Decide on which device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.init()

print(f"{device} is used to train model")
print(f"Use label projection: {use_lp}")
print(f"Use embedding network: {use_embedding_net}")
print(f"Use conditional batch normalization: {use_cbn}")
# Create models
critic = Critic(CHANNELS, target_channels=variable.size(), image_size=IMAGE_SIZE, features=FEATURES,
                lp=use_lp, embed=use_embedding_net, dropout_rate=DROPOUT_RATE).to(device)
if model_type == "dc":
    generator = Generator(CHANNELS, target_channels=variable.size(), y_dim=Y_DIM, z_dim=Z_DIM, image_size=IMAGE_SIZE,
                          features=FEATURES, use_cbn=use_cbn, dropout_rate=DROPOUT_RATE).to(device)
else:
    generator = Generator(CHANNELS, target_channels=variable.size(), y_dim=Y_DIM, z_dim=Z_DIM,
                          image_size=IMAGE_SIZE, features=FEATURES, dropout_rate=DROPOUT_RATE).to(device)

# Initialize weights
initialize_dcgan_weights(critic)
initialize_dcgan_weights(generator)

# Count parameters
critic_params = count_parameters(critic)
gen_params = count_parameters(generator)
print(f"Generator parameters: {gen_params}")
print(f"Critic parameters: {critic_params}")

# Setup optimizers
critic_optim = Adam(critic.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
generator_optim = Adam(generator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))

# Optional feed forward network to use as evaluation metric
forward_network = None
if settings["feed_forward_network"] is not None:
    checkpoint = torch.load(settings["feed_forward_network"], map_location=device)
    try:
        if variable == DimerVariable.ALL:
            forward_network = EfficientNetRegressionV3General(CHANNELS, Y_DIM, activation=Softplus(),
                                                              image_size=IMAGE_SIZE, out_channels=4,
                                                              dropout_rate=0.5).to(device)
            forward_network.load_state_dict(checkpoint['model_state_dict'])
            print("Succesfully loaded forward network as EfficientNet for all dimer variables!")
        elif variable == DimerVariable.CROSS_SECTIONS:
            forward_network = EfficientNetRegressionV3General(CHANNELS, Y_DIM, activation=Softplus(),
                                                              image_size=IMAGE_SIZE, out_channels=2,
                                                              dropout_rate=0.5).to(device)
            forward_network.load_state_dict(checkpoint['model_state_dict'])
            print("Succesfully loaded forward network as EfficientNet for all dimer variables!")
        else:
            forward_network = EfficientNetRegressionV3(CHANNELS, Y_DIM, activation=Softplus(),
                                                       image_size=IMAGE_SIZE, dropout_rate=0.5).to(device)
            forward_network.load_state_dict(checkpoint['model_state_dict'])
            print("Succesfully loaded forward network as EfficientNet!")
    except Exception:
        print("Could not load given forward network as EfficientNet, tries to load as EfficientNetV2...")
        try:
            if variable == DimerVariable.ALL:
                forward_network = EfficientNetV2RegressionGeneral(CHANNELS, Y_DIM, activation=Softplus(),
                                                                  image_size=IMAGE_SIZE, out_channels=4,
                                                                  dropout_rate=0.5).to(device)
                forward_network.load_state_dict(checkpoint['model_state_dict'])
                print("...Succesfully loaded forward network as EfficientNetV2 for all dimer variables!")
            elif variable == DimerVariable.CROSS_SECTIONS:
                forward_network = EfficientNetV2RegressionGeneral(CHANNELS, Y_DIM, activation=Softplus(),
                                                                  image_size=IMAGE_SIZE, out_channels=2,
                                                                  dropout_rate=0.5).to(device)
                forward_network.load_state_dict(checkpoint['model_state_dict'])
                print("Succesfully loaded forward network as EfficientNetV2 for cross section dimer variables!")
            else:
                forward_network = EfficientNetV2Regression(CHANNELS, Y_DIM, activation=Softplus(),
                                                           image_size=IMAGE_SIZE, dropout_rate=0.5).to(device)
                forward_network.load_state_dict(checkpoint['model_state_dict'])
                print("...Succesfully loaded forward network as EfficientNetV2!")
        except Exception:
            raise Exception("The feed forward network is not of type EffNetV2 or EffNet. Can also be that variable is not supported.")

# Train model
save_model_filename = settings["save_model_filename"]
load_model_filename = settings["load_model_filename"]
gan_trainer = WGANTrainer(critic, generator, critic_optim, generator_optim, training_loader, validation_loader, device,
                          load_model_filename=load_model_filename, save_model_filename=save_model_filename,
                          forward_network=forward_network, write_to_tensorboard=settings["tensorboard"])
gan_trainer.train_model(EPOCHS)
