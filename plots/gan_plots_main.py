from pathlib import Path

from plots import GANPlotter
from src.cnn_regression import EfficientNetV2RegressionGeneral
from src.utils import DimerDataset as Dataset
from src.utils import DimerVariable

#################### Dimer cylinders ###################
"""
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
    inverse_transform=lambda x: train_dataset.apply_inverse_transform(x),
    inverse_target_transform=lambda x: train_dataset.apply_inverse_target_transform(x),
)
test_dataset = Dataset(
    test_dir,  # Validation set uses same transforms as in the training set
    DimerVariable.CROSS_SECTIONS,
    transform=lambda x: train_dataset.apply_transform(x),
    target_transform=lambda x: train_dataset.apply_target_transform(x),
    inverse_transform=lambda x: train_dataset.apply_inverse_transform(x),
    inverse_target_transform=lambda x: train_dataset.apply_inverse_target_transform(x),
)

forward_network = {
    "load_path": "delivery/pretrained_cnn_models/last_epoch_effv2_cross_dimer_cylinders_lr00001_drop05.pth.tar",
    "model": EfficientNetV2RegressionGeneral,
}

fcgan_files = {
    "FCGAN": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=0_em=0_data=cyl_drop=0.0_feat=4_last_epoch_20000.pth.tar",
        0.0,
    ),
    "FCGAN + LP": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=1_em=0_data=cyl_drop=0.0_feat=4_last_epoch_20000.pth.tar",
        0.0,
    ),
    "FCGAN + Em.": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=0_em=1_data=cyl_drop=0.0_feat=4_last_epoch_20000.pth.tar",
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
    "DCGAN + Em.": (
        "delivery/aip_review_results/dimer_cylinders/dcgan_lp=0_em=1_data=cyl_drop=0.0_last_epoch_20000.pth.tar",
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
    forward_network,
    train_dataset,
    val_dataset,
    test_dataset,
    fcgan_feature_scaling=4,
    savefig_dir="./figures/aip_review_changes/dimer_cylinders/",
)
gan_plotter.plot_training_and_validation_error()
gan_plotter.plot_error_estimates()

fcgan_files = {
    "FCGAN": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=0_em=0_data=cyl_drop=0.0_feat=4_best_epoch_14901.pth.tar",
        0.0,
    ),
    "FCGAN + LP": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=1_em=0_data=cyl_drop=0.0_feat=4_best_epoch_2901.pth.tar",
        0.0,
    ),
    "FCGAN + LP + Em.": (
        "delivery/aip_review_results/dimer_cylinders/fcgan_lp=1_em=1_data=cyl_drop=0.0_feat=4_best_epoch_2901.pth.tar",
        0.0,
    ),
}
dcgan_files = {
    "DCGAN": (
        "delivery/aip_review_results/dimer_cylinders/dcgan_lp=0_em=0_data=cyl_drop=0.0_best_epoch_6701.pth.tar",
        0.0,
    ),
    "DCGAN + LP": (
        "delivery/aip_review_results/dimer_cylinders/dcgan_lp=1_em=0_data=cyl_drop=0.0_best_epoch_5001.pth.tar",
        0.0,
    ),
    "DCGAN + LP + Em.": (
        "delivery/aip_review_results/dimer_cylinders/dcgan_lp=1_em=1_data=cyl_drop=0.0_best_epoch_4201.pth.tar",
        0.0,
    ),
}
gan_plotter = GANPlotter(
    fcgan_files,
    dcgan_files,
    forward_network,
    train_dataset,
    val_dataset,
    test_dataset,
    fcgan_feature_scaling=4,
    savefig_dir="./figures/aip_review_changes/dimer_cylinders/",
)
gan_plotter.plot_images()
gan_plotter.plot_single_sample_prediction(
    fcgan_keys=["FCGAN", "FCGAN + LP + Em."], dcgan_keys=["DCGAN", "DCGAN + LP + Em."]
)
"""

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
    inverse_transform=lambda x: train_dataset.apply_inverse_transform(x),
    inverse_target_transform=lambda x: train_dataset.apply_inverse_target_transform(x),
)
test_dataset = Dataset(
    test_dir,  # Validation set uses same transforms as in the training set
    DimerVariable.CROSS_SECTIONS,
    transform=lambda x: train_dataset.apply_transform(x),
    target_transform=lambda x: train_dataset.apply_target_transform(x),
    inverse_transform=lambda x: train_dataset.apply_inverse_transform(x),
    inverse_target_transform=lambda x: train_dataset.apply_inverse_target_transform(x),
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

gan_plotter = GANPlotter(
    fcgan_files,
    dcgan_files,
    forward_network,
    train_dataset,
    val_dataset,
    test_dataset,
    fcgan_feature_scaling=4,
    savefig_dir="./figures/aip_review_changes/all_structures/",
)
"""
gan_plotter.plot_training_and_validation_error()
gan_plotter.plot_error_estimates()
"""
fcgan_files = {
    "FCGAN": (
        "delivery/aip_review_results/all_structures/fcgan_lp=0_em=0_data=all_drop=0.0_feat=4_last_epoch_20000.pth.tar",
        0.0,
    ),
    "FCGAN + LP": (
        "delivery/aip_review_results/all_structures/fcgan_lp=1_em=0_data=all_drop=0.0_feat=4_best_epoch_1301.pth.tar",
        0.0,
    ),
    "FCGAN + LP + Em.": (
        "delivery/aip_review_results/all_structures/fcgan_lp=1_em=1_data=all_drop=0.0_feat=4_best_epoch_1801.pth.tar",
        0.0,
    ),
}
dcgan_files = {
    "DCGAN": (
        "delivery/aip_review_results/all_structures/dcgan_lp=0_em=0_data=all_drop=0.0_best_epoch_9801.pth.tar",
        0.0,
    ),
    "DCGAN + LP": (
        "delivery/aip_review_results/all_structures/dcgan_lp=1_em=0_data=all_drop=0.0_best_epoch_1701.pth.tar",
        0.0,
    ),
    "DCGAN + LP + Em.": (
        "delivery/aip_review_results/all_structures/dcgan_lp=1_em=1_data=all_drop=0.0_best_epoch_1601.pth.tar",
        0.0,
    ),
}
gan_plotter = GANPlotter(
    fcgan_files,
    dcgan_files,
    forward_network,
    train_dataset,
    val_dataset,
    test_dataset,
    fcgan_feature_scaling=4,
    savefig_dir="./figures/aip_review_changes/all_structures/",
)
"""
gan_plotter.plot_images()
gan_plotter.plot_prediction_comparison(
    fcgan_labels=["FCGAN", "FCGAN + LP", "FCGAN + LP\n+ Em."],
    dcgan_labels=["DCGAN", "DCGAN + LP", "DCGAN + LP\n+ Em."],
)
"""
gan_plotter.plot_multiple_predictions()
"""
gan_plotter.plot_gaussian_predictions(dcgan_keys=["DCGAN + LP + Em."])
"""
