import yaml

settings = {}
settings["target_variable"] = "cross"
settings["epochs"] = 20000
settings["batch_size"] = 64
settings["learning_rate"] = 0.0001
settings["beta1"] = 0.0
settings["beta2"] = 0.9
settings["feature_scaling"] = 1
settings["tensorboard"] = False
settings["use_cbn"] = False
settings["load_model_filename"] = None

arch = ["fc", "dc"]
dropout_rate = [0.0, 0.5]
feed_forward_network = [
    "delivery/pretrained_cnn_models/last_epoch_effv2_cross_dimer_cylinders_lr00001_drop05.pth.tar",
    "delivery/pretrained_cnn_models/last_epoch_effv2_cross_all_structures_lr00001_drop05.pth.tar",
]
data_dir = [
    "data/dimer_cylinder_train_val_test/",
    "data/anisotropic_au_structures_train_val_test/",
]
use_label_projection = [False, True]
use_embedding_network = [False, True]

base_path = "settings/aip_revision_settings/"

for dr in dropout_rate:
    settings["dropout_rate"] = dr

    for ar in arch:
        settings["architecture"] = ar

        for j, dd in enumerate(data_dir):
            settings["data_dir"] = dd
            settings["feed_forward_network"] = feed_forward_network[j]
            data = "cyl" if (j == 0) else "all"

            for lp in use_label_projection:
                settings["use_label_projection"] = lp

                for em in use_embedding_network:
                    settings["use_embedding_network"] = em
                    filename = (
                        ar
                        + "gan_"
                        + f"lp={int(lp)}_"
                        + f"em={int(em)}_"
                        + f"data={data}_"
                        + f"drop={dr}"
                    )
                    settings["save_model_filename"] = "tmp/" + filename + ".pth.tar"

                    with open(base_path + filename + ".yaml", "w") as file:
                        yaml.dump(settings, file, sort_keys=False)
