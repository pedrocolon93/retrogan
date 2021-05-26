import os.path

import torch.cuda
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from rcgan_pytorch import RetroCycleGAN

local_dir = "/home/pedro/ssd/rg_clean"
def trainable(config):
    # Initialize the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"
    rcgan = RetroCycleGAN(
        save_folder=None,
        generator_lr=config["g_lr"],
        discriminator_lr=config["d_lr"],
        one_way_mm=config["one_way_mm"],
        cycle_mm=config["cycle_mm"],
        cycle_dis=config["cycle_dis"],
        id_loss=config["id_loss"],
        cycle_loss=config["cycle_loss"],
        name=None,
        fp16=fp16
    )
    rcgan.to_device(device)
    print("Runnning on",device)
    ds = {
            "original":os.path.join(local_dir,"Data/ft_all_unseen.txt"),  # "ft_nb_seen.h5",
            "retrofitted": os.path.join(local_dir,"Data/ft_all_retrofitted_ook_unseen.txt"),  # "nb_retrofitted_ook_attractrepel.h5",
        }
    # Save initial version
    rcgan.train_model(epochs=35, batch_size=config["batch_size"], dataset=ds,

                               save_folder=rcgan.save_folder,
                               epochs_per_checkpoint=None,
                               dis_train_amount=config["dis_train_amount"],
                               iters=None,ray=True,wdb=False,tb=False,local_dir=local_dir)
if __name__ == '__main__':
    # Best config:  {'g_lr': 0.00044, 'd_lr': 0.03606, 'one_way_mm': True, 'cycle_mm': True, 'cycle_dis': False, 'id_loss': False, 'cycle_loss': False, 'batch_size': 32, 'generator_size': 2048, 'discriminator_size': 2048, 'generator_hidden_layers': 1, 'discriminator_hidden_layers': 3, 'dis_train_amount': 3}
    config = {
        "g_lr" : tune.qloguniform(0.00005,.1,0.00005),
        "d_lr" : tune.qloguniform(0.00005,.1,0.00005),
        "one_way_mm" : True,
        "cycle_mm" :True,
        "cycle_dis" : True,
        "id_loss" :True,
        "cycle_loss" : True,
        "batch_size":tune.choice([16,32,64]),
        "generator_size": tune.choice([512,1024,2048]),
        "discriminator_size": tune.choice([512,1024,2048]),
        "generator_hidden_layers": tune.choice([1,2,3]),
        "discriminator_hidden_layers": tune.choice([1,2,3]),
        "dis_train_amount":tune.choice([1,2,3])
    }
    score_to_track = "simverb"
    hyperband = ASHAScheduler(metric=score_to_track, mode="max")
    # num_samples = 10

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 4,
            "gpu": 0.25
        },
        scheduler=hyperband,
        num_samples=25,
        config=config,
        name="tune_rg"
    )
    print("Best config: ", analysis.get_best_config(
        metric=score_to_track, mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    df.to_csv("ray_results.csv")
