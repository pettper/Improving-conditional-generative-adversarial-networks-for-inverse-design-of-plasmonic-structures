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

# Configure command line arguments
parser = argparse.ArgumentParser(prog='wgan_gp_main.py', description="The program trains a Wasserstein GAN model on user"
                                                                     + " specified data for a given number of epochs.")
parser.add_argument("-a", "--architecture", type=str, default="fc", help="Model architecture to use.")
parser.add_argument("-var", "--variable", type=str, default="rot", help="Variable from dimer dataset to use.")
parser.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs to train.")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate for training.")
parser.add_argument("-b1", "--beta1", type=float, default=0.0, help="Beta 1 for Adam.")
parser.add_argument("-b2", "--beta2", type=float, default=0.9, help="Beta 2 for Adam.")
parser.add_argument("-sf", "--save_model_filename", type=str, default=None, help="Optional filename to save model.")
parser.add_argument("-lf", "--load_model_filename", type=str, default=None, help="Optional filename to load model.")
parser.add_argument("-ffn", "--feed_forward_network", type=str, default=None,
                    help="Optional feed forward network to use as evaluation metric")
parser.add_argument("-ddir", "--data_dir", type=str, help="Root directory of dataset.")
parser.add_argument("-tb", "--tensorboard", type=bool, default=False, help="Write training results to tensorboard.")
parser.add_argument("-cbn", "--use_cbn", type=str, default="False", help="Use conditional batch normalization.")
parser.add_argument("-lp", "--use_label_projection", type=str, default="True", help="Use label projection.")
parser.add_argument("-em", "--use_embedding_network", type=str, default="True", help="Use embedding network.")
# Parse given command line arguments
args = parser.parse_args()

# Import desired model
model_type = args.architecture
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

# Parameters
CHANNELS = 2
Y_DIM = 81
IMAGE_SIZE = 128
Z_DIM = 100

# Hyperparameters according to WGAN-paper
LEARNING_RATE = args.learning_rate
B1 = args.beta1
B2 = args.beta2
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

# Setup dataloader
# "data/dimer_cylinder_train_val_test/" for example. Expects to find training, validation, and test directories
variable = DimerVariable.keymap(args.variable)
root_dir = Path(args.data_dir)
train_dir = root_dir.joinpath(Path("training/featherfiles/"))
val_dir = root_dir.joinpath(Path("validation/featherfiles/"))

training_dataset = Dataset(train_dir, variable)
validation_dataset = Dataset(val_dir,  # Validation set uses same transforms as in the training set
                             variable,
                             transform=lambda x: training_dataset.apply_transform(x),
                             target_transform=lambda x: training_dataset.apply_target_transform(x))

training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(training_dataset),
                             pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(validation_dataset),
                               pin_memory=True)

# Model
if args.use_label_projection == "True":
    use_lp = True
else:
    use_lp = False
if args.use_cbn == "True":
    use_cbn = True
else:
    use_cbn = False
if args.use_embedding_network == "True":
    use_embedding_net = True
else:
    use_embedding_net = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device} is used to train model")
print(f"Use label projection: {use_lp}")
print(f"Use embedding network: {use_embedding_net}")
print(f"Use conditional batch normalization: {use_cbn}")
# Create models
critic = Critic(CHANNELS, target_channels=variable.size(), image_size=IMAGE_SIZE,
                lp=use_lp, embed=use_embedding_net).to(device)
if model_type == "dc":
    generator = Generator(CHANNELS, target_channels=variable.size(), y_dim=Y_DIM, z_dim=Z_DIM, image_size=IMAGE_SIZE,
                          use_cbn=use_cbn).to(device)
else:
    generator = Generator(CHANNELS, target_channels=variable.size(), y_dim=Y_DIM, z_dim=Z_DIM,
                          image_size=IMAGE_SIZE).to(device)

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
if args.feed_forward_network is not None:
    checkpoint = torch.load(args.feed_forward_network, map_location=device)
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
save_model_filename = args.save_model_filename
load_model_filename = args.load_model_filename
gan_trainer = WGANTrainer(critic, generator, critic_optim, generator_optim, training_loader, validation_loader, device,
                          load_model_filename=load_model_filename, save_model_filename=save_model_filename,
                          forward_network=forward_network, write_to_tensorboard=args.tensorboard)
gan_trainer.train_model(EPOCHS)
