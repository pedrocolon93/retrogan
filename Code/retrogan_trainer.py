import argparse
import os
import random
import shutil

import torch

import helpertools
from rcgan_pytorch import RetroCycleGAN
import numpy as np

def str2bool(v):
    '''Simple helper function to convert an argument to a boolean'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    '''Simple helper function to seed everything'''
    print("Setting seed to", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Terminal options to train RetroGAN.')
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Whether to use fp16 calculation speed up.")
    parser.add_argument("original", default="../Data/ft_all_unseen.txt",
                        help="The input file of the original embeddings. These can be seen as the distributional ones. "
                             "Usually these are fasttext or glove embeddings.")
    parser.add_argument("retrofitted", default="../Data/ft_all_unseen_retrofitted.txt",
                        help="The input file of the retrofitted embeddings. These are the 'specialized' embeddings or the resulting ones after a retrofitting operation."
                             "These are retrofitted counterparts to the original embeddings "
                             "(e.g. after attract-repel of the embeddings passed in 'original')")
    parser.add_argument("model_name",
                        help="The name of the model that we will train.")
    parser.add_argument("save_folder",
                        help="Location where the model will be saved to.")
    parser.add_argument("--epochs_per_checkpoint", default=4, type=int,
                        help="The amount of epochs per checkpoint saved.")
    parser.add_argument("--epochs", default=50, type=int, help="Amount of epochs")
    parser.add_argument("--iters", default=None, type=int, help="Amount of iterations, overrides epochs")

    parser.add_argument("--g_lr", default=0.00005, type=float, help="Generator learning rate")
    parser.add_argument("--d_lr", default=0.0001, type=float, help="Discriminator learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--dis_train_amount", default=3, type=int,
                        help="The amount of times to run a discriminator through the batch")

    parser.add_argument("--one_way_mm", type=str2bool, default=True,
                        help="Whether to enable the one way max margin loss. Enabled by default.")
    parser.add_argument("--cycle_mm", type=str2bool, default=True,
                        help="Whether to enable the cycle max margin loss. Enabled by default.")
    parser.add_argument("--cycle_dis", type=str2bool, default=True,
                        help="Whether to use the Cycle conditional discriminator loss. Enabled by default.")
    parser.add_argument("--id_loss", type=str2bool, default=True,
                        help="Whether to enable the identity loss. This loss is intended to relax a generation if the input is already in the correct domain. Used in CycleGAN. Enabled by default.")
    parser.add_argument("--cycle_loss", type=str2bool, default=True,
                        help="Whether to enable the cycle loss. This is an L1 of X,X' in X->Y->Y->X'. Enabled by default.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed to use for the experiment.")
    parser.add_argument("--device",default="cpu",help="The device to use for training e.g. cuda or cuda:0 or cpu")
    args = parser.parse_args()
    # Set the seed for everything that we use
    set_seed(args.seed)

    test_ds = [
        {
            "original": args.original,  # "ft_nb_seen.h5",
            "retrofitted": args.retrofitted,  # "nb_retrofitted_ook_attractrepel.h5",
            "model_name": args.model_name,  # "Data/nb_ook/",
        }
    ]
    print("Testing arguments:")
    print(args)
    print("\nChecking that everything exists")
    # Check that both of the input data files actually exist!
    for ds in test_ds:
        a = os.path.exists(os.path.join(ds["original"]))
        b = os.path.exists(os.path.join(ds["retrofitted"]))
        print(a, b)
        if not a:
            raise FileNotFoundError("Original file not found " + str(os.path.join(ds["original"])))
        if not b:
            raise FileNotFoundError("Retrofitted file not found" + str(os.path.join(ds["retrofitted"])))

    models = []
    results = []
    save_folder = args.save_folder
    # For every test dataset, train a model
    for idx, ds in enumerate(test_ds):
        # Make the output directory
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        print("Training for dataset:")
        print(ds)
        # Initialize the model
        rcgan = RetroCycleGAN(
            save_folder=args.save_folder,
            generator_lr=args.g_lr,
            discriminator_lr=args.d_lr,
            one_way_mm=args.one_way_mm,
            cycle_mm=args.cycle_mm,
            cycle_dis=args.cycle_dis,
            id_loss=args.id_loss,
            cycle_loss=args.cycle_loss,
            name=args.model_name,
            fp16=args.fp16
        )
        # Get the amount of parameters that are trained
        model_parameters = filter(lambda p: p.requires_grad, rcgan.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Parameters in the whole model:", params)
        model_parameters = filter(lambda p: p.requires_grad, rcgan.g_AB.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Parameters in the generator model:", params)
        model_parameters = filter(lambda p: p.requires_grad, rcgan.d_A.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Parameters in the discriminator model:", params)
        model_parameters = filter(lambda p: p.requires_grad, rcgan.d_ABBA.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Parameters in the cycle discriminator model:", params)
        # If continuing training uncomment this
        # rcgan.load_weights(preface="final", folder="trained_models/retrogans/ft_nb_retrogan/")

        # Send to cpu for evaluation
        rcgan.to_device('cpu')
        # Get initial testing results to see we are starting from scratch.
        # Load the data
        X_train, Y_train = helpertools.load_all_words_dataset_final(ds["original"], ds["retrofitted"],
                                                                         save_folder=save_folder, cache=False)
        sl_start = helpertools.test_original_vectors(X_train, dataset_location="../Data/testing/simlexorig999.txt", prefix="en_")
        sv_start = helpertools.test_original_vectors(X_train, dataset_location="../Data/testing/simverb3500.txt", prefix="en_")
        c_start = helpertools.test_original_vectors(X_train, dataset_location="../Data/testing/card660.tsv", prefix="en_")
        sl_rstart = helpertools.test_original_vectors(Y_train, dataset_location="../Data/testing/simlexorig999.txt",
                                                           prefix="en_")
        sv_rstart = helpertools.test_original_vectors(Y_train, dataset_location="../Data/testing/simverb3500.txt", prefix="en_")
        c_rstart = helpertools.test_original_vectors(Y_train, dataset_location="../Data/testing/card660.tsv", prefix="en_")
        # Bring back to device for testing!
        rcgan.to_device(args.device)

        print("For simlex:", "distributional:", float(sl_start), "retrofitted:", float(sl_rstart))
        print("For simverb:", "distributional:", float(sv_start), "retrofitted:", float(sv_rstart))
        print("For card:", "distributional:", float(c_start), "retrofitted:", float(c_rstart))


        # Save initial version
        ds_res = rcgan.train_model(epochs=args.epochs, batch_size=args.batch_size, dataset=ds,
                                   save_folder=rcgan.save_folder,
                                   epochs_per_checkpoint=args.epochs_per_checkpoint,
                                   dis_train_amount=args.dis_train_amount,
                                   iters=args.iters)
        results.append(ds_res)
        print("*" * 100)
        print(ds, results[-1])
        print("*" * 100)
        print("Saving")
        model_save_folder = os.path.join(args.save_folder, str(idx))
        os.makedirs(model_save_folder, exist_ok=True)
        with open(os.path.join(model_save_folder, "config"), "w") as f:
            f.write('\n------Dataset-------\n')
            f.write(str(ds))
            f.write('\n------Arguments-------\n')
            f.write(str(args))
        with open(os.path.join(model_save_folder, "results"), "w") as f:
            f.write(str(results[-1]))
        models[-1].save_folder = model_save_folder
        models[-1].save_model("final")
        print("Done")
