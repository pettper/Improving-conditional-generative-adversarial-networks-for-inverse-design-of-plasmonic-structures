from plots import GANPlotter

fcgan_files = {
    "FCGAN + LP + Embed. with dropout": "delivery/aip_review_results/fcgan_lp=1_em=1_data=all_drop=0.5_last_epoch_30000.pth.tar",
    "FCGAN + LP + Embed. no dropout": "delivery/aip_review_results/fcgan_lp=1_em=1_data=all_drop=0.0_last_epoch_30000.pth.tar",
}

dcgan_files = {
    "DCGAN + LP + Embed. with dropout": "delivery/aip_review_results/dcgan_lp=1_em=1_data=all_drop=0.5_last_epoch_30000.pth.tar",
    "DCGAN + LP + Embed. no dropout": "delivery/aip_review_results/dcgan_lp=1_em=1_data=all_drop=0.0_best_epoch_14151.pth.tar",
}

gan_plotter = GANPlotter(
    fcgan_files, dcgan_files, savefig_dir="./figures/aip_review_changes/"
)
gan_plotter.plot()
