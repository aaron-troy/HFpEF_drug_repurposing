
import data_loader as dl
import autoencoder
import trainer

def main(args):
    # Construct data loaders
    path = "input_data/GSE92742_Level5_isgold_drugs_lm_hf_modZ.csv"
    train_loader, test_loader = dl.build_loaders(path, in_format = 'z_score')

    latent_sizes = [1100]#[700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    for ls in latent_sizes:

        # Build the architecture
        AE = autoencoder.Autoencoder(960, ls)
        ID = 'leakyReLu_960_' + str(ls)

        # Training
        model_loss = trainer.train_AE(AE, train_loader, test_loader, method='Adam', learning_rate=1e-5, stop_threshold=1e-15, n_epochs=1000, max_patience=20, train_id=ID)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = False
    main(args)

