import argparse
from pathlib import Path
from src.cnn_regression import ModelTrainer
from src.utils import DimerVariable, DimerDataset as Dataset
from src.utils import count_parameters
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss, Softplus

# Configure command line arguments
parser = argparse.ArgumentParser(prog='cnn_main.py', description="The program runs a CNN regression model on user"
                                 + " specified data for a given number of epochs.")
parser.add_argument("-a", "--architecture", type=str, default="eff", help="Architecture to use in model.")
parser.add_argument("-var", "--variable", type=str, default="rot", help="Variable from dimer dataset to use.")
parser.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs to train.")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate for training.")
parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="Beta 1 for Adam.")
parser.add_argument("-b2", "--beta2", type=float, default=0.99, help="Beta 2 for Adam.")
parser.add_argument("-sf", "--save_model_filename", type=str, default=None, help="Optional filename to save model.")
parser.add_argument("-lf", "--load_model_filename", type=str, default=None, help="Optional filename to load model.")
parser.add_argument("-tb", "--tensorboard", type=bool, default=False, help="Write training results to tensorboard.")
parser.add_argument("-ddir", "--data_dir", type=str, help="Root directory of dataset.")
parser.add_argument("-dr", "--drop_out", type=float, default=0.5, help="Dropout probability")
# Parse given command line arguments
args = parser.parse_args()

# Model type
model_type = args.architecture
variable = DimerVariable.keymap(args.variable)
if model_type == "eff":
    if variable == DimerVariable.ALL or variable == DimerVariable.CROSS_SECTIONS:
        from src.cnn_regression import EfficientNetRegressionV3General as RegressionModel
    else:
        from src.cnn_regression import EfficientNetRegressionV3 as RegressionModel
elif model_type == "effv2":
    if variable == DimerVariable.ALL or variable == DimerVariable.CROSS_SECTIONS:
        from src.cnn_regression import EfficientNetV2RegressionGeneral as RegressionModel
    else:
        from src.cnn_regression import EfficientNetV2Regression as RegressionModel
else:
    raise Exception("No model architecture was specified.")


# Set seed
torch.manual_seed(23)

# Hyperparameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
B1 = args.beta1
B2 = args.beta2
DROP_OUT = args.drop_out

# Parameters
IMAGE_SIZE = 128
INPUT_CHANNELS = 2
Y_DIM = 81

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device} is used to train model")
if variable == DimerVariable.ALL:
    model = RegressionModel(INPUT_CHANNELS, Y_DIM, activation=Softplus(), image_size=IMAGE_SIZE,
                            dropout_rate=DROP_OUT, out_channels=4).to(device)
elif variable == DimerVariable.CROSS_SECTIONS:
    model = RegressionModel(INPUT_CHANNELS, Y_DIM, activation=Softplus(), image_size=IMAGE_SIZE,
                            dropout_rate=DROP_OUT, out_channels=2).to(device)
else:
    model = RegressionModel(INPUT_CHANNELS, Y_DIM, activation=Softplus(), image_size=IMAGE_SIZE,
                            dropout_rate=DROP_OUT).to(device)
n_params = count_parameters(model)
print(f"Number of model parameters: {n_params}")

# Setup dataloader
# "data/dimer_cylinder_train_val_test/" for example. Expects to find training, validation, and test directories
root_dir = Path(args.data_dir)
train_dir = root_dir.joinpath(Path("training/featherfiles/"))
val_dir = root_dir.joinpath(Path("validation/featherfiles/"))

training_dataset = Dataset(train_dir, variable)
validation_dataset = Dataset(val_dir,  # Validation set uses same transforms as in the training set
                             variable,
                             transform=lambda x: training_dataset.apply_transform(x),
                             target_transform=lambda x: training_dataset.apply_target_transform(x))

training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(training_dataset),
                             pin_memory=True, drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(validation_dataset),
                               pin_memory=True)

# Define loss, optimizer
loss = MSELoss()
metric = L1Loss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, betas=(B1, B2))

# Setup model trainer
save_model_filename = args.save_model_filename
load_model_filename = args.load_model_filename
model_trainer = ModelTrainer(model, optimizer, loss, training_loader, validation_loader, device, metric=metric,
                             load_model_filename=load_model_filename, save_model_filename=save_model_filename,
                             write_to_tensorboard=args.tensorboard)

# Run
model_trainer.train_model(EPOCHS)
