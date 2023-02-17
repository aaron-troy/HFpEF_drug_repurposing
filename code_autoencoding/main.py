
import data_loader as dl
import autoencoder
import trainer

def main(args):
    # Construct data loaders
    path = "../data/shared_landmark_counts_vecs.gctx_n1269922x962.gctx"
    data_format = 'counts'
    train_loader, test_loader = dl.build_loaders(path, in_format = data_format)

    latent_sizes = [1024] #[896, 1024, 1152, 1280, 1408, 1536]

    for ls in latent_sizes:

        # Build the architecture
        AE = autoencoder.Autoencoder(962, ls)
        ID = data_format + '150epoch_leakyReLu_962_' + str(ls)

        # Training
        model_loss = trainer.train_AE(AE, train_loader, test_loader, method='Adam', learning_rate=1e-4, stop_threshold=1e-15, n_epochs=150, max_patience=10, train_id=ID)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = False
    main(args)

