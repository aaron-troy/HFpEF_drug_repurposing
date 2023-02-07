
import data_loader as dl
import autoencoder
import trainer

def main(args):
    # Construct data loaders
    path = "../data/shared_landmark_counts_vecs.gctx_n100000x960.gct"
    data_format = 'counts'
    train_loader, test_loader = dl.build_loaders(path, in_format = data_format)

    latent_sizes = [700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    for ls in latent_sizes:

        # Build the architecture
        AE = autoencoder.Autoencoder(960, ls)
        ID = data_format + '_leakyReLu_960_' + str(ls)

        # Training
        model_loss = trainer.train_AE(AE, train_loader, test_loader, method='Adam', learning_rate=1e-4, stop_threshold=1e-15, n_epochs=1000, max_patience=20, train_id=ID)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = False
    main(args)

