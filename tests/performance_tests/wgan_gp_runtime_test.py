import argparse

from src.utils import load_yaml
from wgan_gp_main import WGANGPMain

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

main.run(benchmark=True)
