from pathlib import Path

from torch.utils.data import DataLoader, RandomSampler

from plots import GANPlotter
from src.utils import DimerDataset as Dataset
from src.utils import DimerVariable

#################### Dimer cylinders ###################
root_dir = Path("./data/dimer_cylinder_train_val_test")
train_dir = root_dir.joinpath(Path("training/featherfiles/"))
val_dir = root_dir.joinpath(Path("validation/featherfiles/"))
test_dir = root_dir.joinpath(Path("test/featherfiles/"))

train_dataset = Dataset(train_dir, DimerVariable.CROSS_SECTIONS)
val_dataset = Dataset(
    val_dir,  # Validation set uses same transforms as in the training set
    DimerVariable.CROSS_SECTIONS,
    transform=lambda x: train_dataset.apply_transform(x),
    target_transform=lambda x: train_dataset.apply_target_transform(x),
)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=RandomSampler(train_dataset),
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    sampler=RandomSampler(val_dataset),
    pin_memory=True,
)

fcgan_files = {
    "FCGAN": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=0_em=0_data=cyl_drop=0.0_feat=4_last_epoch_20000.pth.tar",
        0.0,
    ),
    "FCGAN + LP": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=1_em=0_data=cyl_drop=0.0_feat=4_last_epoch_20000.pth.tar",
        0.0,
    ),
    "FCGAN + LP + Em.": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=1_em=1_data=cyl_drop=0.0_feat=4_last_epoch_20000.pth.tar",
        0.0,
    ),
}

dcgan_files = {
    "DCGAN": (
        "delivery/aip_review_results/dimer_cylinders/dcgan_lp=0_em=0_data=cyl_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
    "DCGAN + LP": (
        "delivery/aip_review_results/dimer_cylinders/dcgan_lp=1_em=0_data=cyl_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
    "DCGAN + LP + Em.": (
        "delivery/aip_review_results/dimer_cylinders/dcgan_lp=1_em=1_data=cyl_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
}

gan_plotter = GANPlotter(
    fcgan_files,
    dcgan_files,
    fcgan_feature_scaling=4,
    savefig_dir="./figures/aip_review_changes/dimer_cylinders/",
)
gan_plotter.plot_training_and_validation_error()
gan_plotter.plot_error_estimates()


fcgan_files = {
    "FCGAN + LP no dropout epoch 8701": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=1_em=0_data=cyl_drop=0.0_best_epoch_8701.pth.tar",
        0.0,
    ),
    "FCGAN + LP no dropout epoch 16501": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=1_em=0_data=cyl_drop=0.5_best_epoch_16501.pth.tar",
        0.0,
    ),
    "FCGAN + LP + Em. no dropout epoch 20000": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=1_em=1_data=cyl_drop=0.5_last_epoch_20000.pth.tar",
        0.0,
    ),
}
dcgan_files = {}
gan_plotter = GANPlotter(
    fcgan_files,
    dcgan_files,
    train_loader=train_loader,
    val_loader=val_loader,
    savefig_dir="./figures/aip_review_changes/dimer_cylinders/",
)
gan_plotter.plot_images()


############# All structures #################
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
)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=RandomSampler(train_dataset),
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    sampler=RandomSampler(val_dataset),
    pin_memory=True,
)

fcgan_files = {
    "FCGAN with dropout": (
        "delivery/aip_review_results/all_structures/fcgan_lp=0_em=0_data=all_drop=0.5_last_epoch_20000.pth.tar",
        0.5,
    ),
    "FCGAN no dropout": (
        "delivery/aip_review_results/all_structures/fcgan_lp=0_em=0_data=all_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
    "FCGAN + LP with dropout": (
        "delivery/aip_review_results/all_structures/fcgan_lp=1_em=0_data=all_drop=0.5_last_epoch_20000.pth.tar",
        0.5,
    ),
    "FCGAN + LP no dropout": (
        "delivery/aip_review_results/all_structures/fcgan_lp=1_em=0_data=all_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
    "FCGAN + LP + Em. with dropout": (
        "delivery/aip_review_results/all_structures/fcgan_lp=1_em=1_data=all_drop=0.5_last_epoch_20000.pth.tar",
        0.5,
    ),
    "FCGAN + LP + Em. no dropout": (
        "delivery/aip_review_results/all_structures/fcgan_lp=1_em=1_data=all_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
}

dcgan_files = {
    "DCGAN with dropout": (
        "delivery/aip_review_results/all_structures/dcgan_lp=0_em=0_data=all_drop=0.5_last_epoch_20000.pth.tar",
        0.5,
    ),
    "DCGAN no dropout": (
        "delivery/aip_review_results/all_structures/dcgan_lp=0_em=0_data=all_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
    "DCGAN + LP with dropout": (
        "delivery/aip_review_results/all_structures/dcgan_lp=1_em=0_data=all_drop=0.5_last_epoch_20000.pth.tar",
        0.5,
    ),
    "DCGAN + LP no dropout": (
        "delivery/aip_review_results/all_structures/dcgan_lp=1_em=0_data=all_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
    "DCGAN + LP + Em. with dropout": (
        "delivery/aip_review_results/all_structures/dcgan_lp=1_em=1_data=all_drop=0.5_last_epoch_20000.pth.tar",
        0.5,
    ),
    "DCGAN + LP + Em. no dropout": (
        "delivery/aip_review_results/all_structures/dcgan_lp=1_em=1_data=all_drop=0.0_last_epoch_20000.pth.tar",
        0.0,
    ),
}

gan_plotter = GANPlotter(
    fcgan_files, dcgan_files, savefig_dir="./figures/aip_review_changes/all_structures/"
)
gan_plotter.plot()

fcgan_files = {
    "FCGAN + LP + Em. no dropout": (
        "delivery/aip_review_results/all_structures/fcgan_lp=1_em=1_data=all_drop=0.0_best_epoch_2001.pth.tar",
        0.0,
    ),
}
dcgan_files = {
    "DCGAN + LP + Em. no dropout": (
        "delivery/aip_review_results/all_structures/dcgan_lp=1_em=1_data=all_drop=0.0_best_epoch_1601.pth.tar",
        0.0,
    ),
}
gan_plotter = GANPlotter(
    fcgan_files,
    dcgan_files,
    train_loader=train_loader,
    val_loader=val_loader,
    savefig_dir="./figures/aip_review_changes/all_structures/",
)
gan_plotter.plot_images()
