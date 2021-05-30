import datetime
import os
import random

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
try:
    from ray import tune
except:
    print("Could not load ray!")
try:
    import helpertools
except:
    print("Could not load helper tools!")


class MaxMargin_Loss(torch.nn.Module):
    '''Max margin loss class.  An adaptation of the one utilized in AuxGAN.  '''

    def __init__(self, sim_neg=25, batch_size=32, sim_margin=1):
        super(MaxMargin_Loss, self).__init__()
        # Amount of times to calculate the loss
        self.sim_neg = sim_neg
        self.batch_size = batch_size
        self.sim_margin = sim_margin

    def forward(self, y_pred, y_true):
        cost = 0.
        for i in range(0, self.sim_neg):
            # Gets a random set from the current batch
            new_true = torch.randperm(self.batch_size).to(y_pred.device)
            new_true = y_true[new_true]
            # Normalize everything for a cosine similarity
            normalize_a = self.l2_norm(y_true)
            normalize_b = self.l2_norm(y_pred)
            normalize_c = self.l2_norm(new_true)
            # Cosine similarity, things in the original batch should be close together, things in the other batch
            # should be further apart
            minimize = torch.sum(torch.multiply(normalize_a, normalize_b))
            maximize = torch.sum(torch.multiply(normalize_a, normalize_c))
            # Actual calculation for the loss
            mg = self.sim_margin - minimize + maximize
            # Clamp it at 0 because it can be negative.
            cost += torch.clamp(mg, min=0)
        # Since we are getting the cost for sim_neg, normalize by dividing by the amount
        return cost / self.sim_neg

    def l2_norm(self, x):
        sq = torch.square(x)
        square_sum = torch.sum(torch.sum(sq, dim=1))
        epsilon = 1e-12
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        normalize_a_t = x * x_inv_norm
        return normalize_a_t


class CycleCond_Loss(torch.nn.Module):
    # Class that implements the conditional gan loss.
    def __init__(self):
        super(CycleCond_Loss, self).__init__()

    def forward(self, d_ground, d_approx):
        cost = torch.log(d_ground) + torch.log(1 - d_approx)
        return -1 * cost.mean()


class RetroCycleGAN(nn.Module):
    '''Main class to represent RetroGAN.'''

    def forward(self, x):
        return self.g_AB(x)

    def __init__(self, save_index="0", save_folder="./", generator_size=2048,
                 generator_hidden_layers = 1, discriminator_hidden_layers = 1,
                 discriminator_size=2048, word_vector_dimensions=300,
                 discriminator_lr=0.0001, generator_lr=0.0001,
                 one_way_mm=True, cycle_mm=True, cycle_dis=True, id_loss=True, cycle_loss=True,
                 device="cpu", name="default", fp16=False):
        super().__init__()
        self.fp16 = fp16
        self.save_folder = save_folder
        self.device = device
        # Input vector length
        self.word_vector_dimensions = word_vector_dimensions
        # Index of model that we are saving. Used for checkpoints.
        self.save_index = save_index
        # Size of the hidden layers
        self.gf = generator_size
        self.df = discriminator_size
        self.g_ha = generator_hidden_layers
        self.d_ha = discriminator_hidden_layers
        # Model name
        self.name = name
        # Set the learning rates
        self.d_lr = discriminator_lr
        self.g_lr = generator_lr
        # Configuration for losses
        self.one_way_mm = one_way_mm
        self.cycle_mm = cycle_mm
        self.cycle_dis = cycle_dis
        self.id_loss = id_loss
        self.cycle_mm_weight = 2
        self.id_loss_weight = 0.01
        self.cycle_loss = cycle_loss

        self.local_dir = "../"
        # Simple trick to not suck up the memory when loading tensorflow.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        # We use the GlorotUniform initializer from Keras.  This gave good performance in a tensorflow implementation.
        self.initializer = tf.keras.initializers.GlorotUniform()

        # Construct the inner workings of the model
        # Discriminators
        self.d_A = self.build_discriminator(hidden_dim=self.df)
        self.d_B = self.build_discriminator(hidden_dim=self.df)
        # Conditional discriminators
        self.d_ABBA = self.build_c_discriminator(hidden_dim=self.df)
        self.d_BAAB = self.build_c_discriminator(hidden_dim=self.df)
        # Generators
        self.g_AB = self.build_generator(hidden_dim=self.gf)
        self.g_BA = self.build_generator(hidden_dim=self.gf)

    def put_train(self):
        '''Function to set internal models as training'''
        self.d_A.train()
        self.d_B.train()
        self.d_ABBA.train()
        self.d_BAAB.train()
        self.g_AB.train()
        self.g_BA.train()

    def put_eval(self):
        '''Function to set internal models as evaluating. Eliminates dropout.'''
        self.d_A.eval()
        self.d_B.eval()
        self.d_ABBA.eval()
        self.d_BAAB.eval()
        self.g_AB.eval()
        self.g_BA.eval()

    def set_fp16(self):
        self.d_A = self.d_A.half() if self.fp16 else self.d_A
        self.d_B = self.d_B.half() if self.fp16 else self.d_B
        self.d_ABBA = self.d_ABBA.half() if self.fp16 else self.d_ABBA
        self.d_BAAB = self.d_BAAB.half() if self.fp16 else self.d_BAAB
        self.g_AB = self.g_AB.half() if self.fp16 else self.g_AB
        self.g_BA = self.g_BA.half() if self.fp16 else self.g_BA
    def set_fp32(self):
        self.d_A = self.d_A.float() if self.fp16 else self.d_A
        self.d_B = self.d_B.float() if self.fp16 else self.d_B
        self.d_ABBA = self.d_ABBA.float() if self.fp16 else self.d_ABBA
        self.d_BAAB = self.d_BAAB.float() if self.fp16 else self.d_BAAB
        self.g_AB = self.g_AB.float() if self.fp16 else self.g_AB
        self.g_BA = self.g_BA.float() if self.fp16 else self.g_BA


    def compile_all(self):
        '''Function that compiles the optimizers for the inner models'''
        # Discriminator optimizers
        self.dA_optimizer = Adam(self.d_A.parameters(), lr=self.d_lr, eps=1e-10)
        self.dB_optimizer = Adam(self.d_B.parameters(), lr=self.d_lr, eps=1e-10)
        # Conditional discriminator optimizers
        self.dABBA_optimizer = Adam(self.d_ABBA.parameters(), lr=self.d_lr, eps=1e-10)
        self.dBAAB_optimizer = Adam(self.d_BAAB.parameters(), lr=self.d_lr, eps=1e-10)
        # Generator optimizers
        self.g_AB_optimizer = Adam(self.g_AB.parameters(), lr=self.g_lr, eps=1e-10)
        self.g_BA_optimizer = Adam(self.g_BA.parameters(), lr=self.g_lr, eps=1e-10)
        # Combined model optimizer for the Cycles
        self.combined_optimizer = Adam([x for x in self.g_BA.parameters()] +
                                       [x for x in self.g_AB.parameters()],
                                       lr=self.g_lr, eps=1e-10)
        # For fp16 use the correct scaler
        if self.fp16:
            self.dA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dABBA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dBAAB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.g_AB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.g_BA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.combined_optimizerscaler = torch.cuda.amp.GradScaler()

    def build_generator(self, hidden_dim=2048, dropout=0.2):
        '''Function to construct the generator model'''
        # Input layer
        inpt = nn.Linear(self.word_vector_dimensions, hidden_dim)
        s = torch.tensor(self.initializer(shape=inpt.weight.shape).numpy())
        inpt.weight = nn.parameter.Parameter(s)
        inpt.bias.data.fill_(0)
        # Output layer
        out = nn.Linear(hidden_dim, self.word_vector_dimensions)
        s = torch.tensor(self.initializer(shape=out.weight.shape).numpy())
        out.weight = nn.parameter.Parameter(s)
        out.bias.data.fill_(0)
        # Build the layers
        layers = []
        layers.append(inpt)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for i in range(self.g_ha):
            # Hidden layer
            hid = nn.Linear(hidden_dim, hidden_dim)
            s = torch.tensor(self.initializer(shape=hid.weight.shape).numpy())
            hid.weight = nn.parameter.Parameter(s)
            hid.bias.data.fill_(0)
            layers.append(hid)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(out)
        # Return a sequential model with all of this.
        return nn.Sequential(
            *layers
        )

    def build_discriminator(self, hidden_dim=2048, dropout=0.3):
        '''Helper funtion to build discriminators'''
        # Input layer
        inpt = nn.Linear(self.word_vector_dimensions, hidden_dim)
        s = torch.tensor(self.initializer(shape=inpt.weight.shape).numpy())
        inpt.weight = nn.parameter.Parameter(s)
        inpt.bias.data.fill_(0)

        # Ouput layer
        out = nn.Linear(hidden_dim, 1)
        s = torch.tensor(self.initializer(shape=out.weight.shape).numpy())
        out.weight = nn.parameter.Parameter(s)
        out.bias.data.fill_(0)
        # Construct a sequential
        layers = []
        layers.append(inpt)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for i in range(self.d_ha):
            # Hidden layer
            hid = nn.Linear(hidden_dim, hidden_dim)
            s = torch.tensor(self.initializer(shape=hid.weight.shape).numpy())
            hid.weight = nn.parameter.Parameter(s)
            hid.bias.data.fill_(0)
            layers.append(hid)
            layers.append(nn.ReLU())
            # Batch norm for stability
            bn = nn.BatchNorm1d(hidden_dim, momentum=0.99, eps=0.001)
            layers.append(bn)
            layers.append(nn.Dropout(dropout))
        layers.append(out)
        # layers.append(nn.Sigmoid()) # We comment this out because we are using BCE with Logits loss and that has this already.)

        return nn.Sequential(
            *layers
        )

    def build_c_discriminator(self, hidden_dim=2048, dropout=0.3):
        '''Helper function to construct the conditional discriminators'''
        # Input layer, note that since we are conditioning on a prior vector input, we duplicate the inputs
        inpt = nn.Linear(self.word_vector_dimensions * 2, hidden_dim)
        s = torch.tensor(self.initializer(shape=inpt.weight.shape).numpy())
        inpt.weight = nn.parameter.Parameter(s)
        inpt.bias.data.fill_(0)
        # Output layer
        out = nn.Linear(hidden_dim, 1)
        s = torch.tensor(self.initializer(shape=out.weight.shape).numpy())
        out.weight = nn.parameter.Parameter(s)
        out.bias.data.fill_(0)

        # Construct a sequential
        layers = []
        layers.append(inpt)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for i in range(self.d_ha):
            # Hidden layer
            hid = nn.Linear(hidden_dim, hidden_dim)
            s = torch.tensor(self.initializer(shape=hid.weight.shape).numpy())
            hid.weight = nn.parameter.Parameter(s)
            hid.bias.data.fill_(0)
            layers.append(hid)
            layers.append(nn.ReLU())
            # Batch norm for stability
            bn = nn.BatchNorm1d(hidden_dim, momentum=0.99, eps=0.001)
            layers.append(bn)
            layers.append(nn.Dropout(dropout))
        layers.append(out)
        layers.append(nn.Sigmoid())

        return nn.Sequential(
            *layers
        )


    def train_model(self, epochs, dataset, save_folder, batch_size=1, cache=False, epochs_per_checkpoint=5,
                    dis_train_amount=3, iters=None,wdb=True, tb=True, ray=False,local_dir="../"):
        self.local_dir = local_dir
        # Make a writer for Tensorboard
        if tb:
            writer = SummaryWriter()
        # Use wandb for watching the model
        if wdb:
            wandb.init(project="retrogan")
            wandb.run.name = self.name
            wandb.watch(self, criterion="simlex")
            wandb.run.save()

        res = []
        self.set_fp16()
        self.to_device(self.device)
        class RetroPairsDataset(Dataset):
            """Dataset of pairs of embeddings consisting of the distributional and its retrofitted counterpart."""

            def __init__(self, original_dataset, retrofitted_dataset, save_folder, cache):
                # Load the data.
                X_train, Y_train = helpertools.load_all_words_dataset_final(original_dataset, retrofitted_dataset,
                                                                                 save_folder=save_folder, cache=cache)
                print("Shapes of training data:",
                      X_train.shape,
                      Y_train.shape)
                print(X_train)
                print(Y_train)
                print("*" * 100)
                self.x = X_train
                self.y = Y_train

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                # We normalize the embeddings that we utilize
                imgs_A = np.array(self.x.iloc[idx], dtype=np.float)
                imgs_B = np.array(self.y.iloc[idx], dtype=np.float)
                imgs_A /= np.linalg.norm(imgs_A)
                imgs_B /= np.linalg.norm(imgs_B)
                return torch.from_numpy(imgs_A), torch.from_numpy(imgs_B)

        # Initialize the dataset
        ds = RetroPairsDataset(dataset["original"], dataset["retrofitted"],
                               save_folder=save_folder, cache=cache)
        # Create our data loader
        dataloader = DataLoader(ds, batch_size=batch_size,
                                shuffle=True, num_workers=0)

        # Initialize our models optimizers
        self.compile_all()

        def train_step(self, batch_i, imgs_A, imgs_B, epoch, count, training_epochs):

                if imgs_A.shape[0] == 1:
                    print("Batch is equal to 1 in training.")
                    return
                a = datetime.datetime.now()
                imgs_A = imgs_A.to(self.device)
                imgs_B = imgs_B.to(self.device)

                imgs_A = imgs_A.half() if self.fp16 else imgs_A.float()
                imgs_B = imgs_B.half() if self.fp16 else imgs_B.float()

                with torch.cuda.amp.autocast():
                    fake_B = self.g_AB(imgs_A)
                    fake_A = self.g_BA(imgs_B)
                # Train the discriminators (original images = real / translated = Fake)
                dA_loss = None
                dB_loss = None
                valid = torch.ones((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                fake = torch.zeros((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                # accs = []
                b = datetime.datetime.now()
                # print("Data prep time",b-a)
                # TRAIN THE DISCRIMINATORS
                a = datetime.datetime.now()
                if False:
                    for _ in range(int(dis_train_amount)):
                        if _ % 2 == 0:
                            # print("Adding noise")
                            i_A = imgs_A + torch.tensor(
                                np.random.uniform(low=-1, size=(imgs_A.shape[0], self.word_vector_dimensions)),
                                device=imgs_A.device).half()
                            i_B = imgs_B + torch.tensor(
                                np.random.uniform(low=-1, size=(imgs_A.shape[0], self.word_vector_dimensions)),
                                device=imgs_B.device).half()
                            f_A = fake_A + torch.tensor(
                                np.random.uniform(low=-1, size=(imgs_A.shape[0], self.word_vector_dimensions)),
                                device=fake_A.device).half()
                            f_B = fake_B + torch.tensor(
                                np.random.uniform(low=-1, size=(imgs_A.shape[0], self.word_vector_dimensions)),
                                device=fake_B.device).half()
                        else:
                            i_A = imgs_A
                            i_B = imgs_B
                            f_B = fake_B
                            f_A = fake_A
                        # with torch.no_grad():
                        # TRAIN ON BATCH VALID
                        self.dA_optimizer.zero_grad()
                        dA = self.d_A(i_A)
                        dA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)

                        if self.fp16:
                            self.dA_optimizerscaler.scale(dA_loss_real).backward()
                            self.dA_optimizerscaler.step(self.dA_optimizer)
                            self.dA_optimizerscaler.update()
                        else:
                            dA_loss_real.backward(retain_graph=True)
                            self.dA_optimizer.step()
                        # TRAIN ON BATCH FAKE
                        self.dA_optimizer.zero_grad()
                        dA_f = self.d_A(f_A)
                        dA_loss_fake = nn.BCEWithLogitsLoss()(dA_f, fake)

                        if self.fp16:
                            self.dA_optimizerscaler.scale(dA_loss_fake).backward(retain_graph=True)
                            self.dA_optimizerscaler.step(self.dA_optimizer)
                            self.dA_optimizerscaler.update()
                        else:
                            dA_loss_fake.backward(retain_graph=True)
                            self.dA_optimizer.step()

                        if dA_loss is None:
                            dA_loss = 0.5 * (float(dA_loss_real) + float(dA_loss_fake))
                        else:
                            dA_loss += 0.5 * (float(dA_loss_real) + float(dA_loss_fake))

                        # TRAIN ON BATCH VALID
                        self.dB_optimizer.zero_grad()
                        dB = self.d_B(i_B)
                        dB_loss_real = nn.BCEWithLogitsLoss()(dB, valid)
                        if self.fp16:
                            self.dB_optimizerscaler.scale(dB_loss_real).backward()
                            self.dB_optimizerscaler.step(self.dB_optimizer)
                            self.dB_optimizerscaler.update()
                        else:
                            dB_loss_real.backward(retain_graph=True)
                            self.dB_optimizer.step()

                        # TRAIN ON BATCH FAKE
                        self.dB_optimizer.zero_grad()
                        dB_f = self.d_B(f_B)
                        dB_loss_fake = nn.BCEWithLogitsLoss()(dB_f, fake)

                        if self.fp16:
                            self.dB_optimizerscaler.scale(dB_loss_fake).backward(retain_graph=True)
                            self.dB_optimizerscaler.step(self.dB_optimizer)
                            self.dB_optimizerscaler.update()
                        else:
                            dB_loss_fake.backward(retain_graph=True)
                            self.dB_optimizer.step()

                        # dB_loss_real = self.d_B.train_on_batch(retrofitted_embeddings, valid)
                        # dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                        if dB_loss is None:
                            dB_loss = 0.5 * (dB_loss_real.item() + dB_loss_fake.item())
                        else:
                            dB_loss += 0.5 * (dB_loss_real.item() + dB_loss_fake.item())
                else:
                    dA_loss = 0
                    dB_loss = 0
                # ABBA
                b = datetime.datetime.now()
                d_loss = (1.0 / dis_train_amount) * 0.5 * np.add(dA_loss, dB_loss)

                # print("Dis train time", b - a)
                # TRAIN THE CYCLE DISCRIMINATORS
                if self.cycle_dis:
                    a = datetime.datetime.now()
                    with torch.cuda.amp.autocast():
                        fake_ABBA = self.g_BA(fake_B)
                        fake_BAAB = self.g_AB(fake_A)
                    self.dABBA_optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        dA = self.d_ABBA(torch.cat([fake_B, imgs_A], 1))
                        dA_r = self.d_ABBA(torch.cat([fake_B, fake_ABBA], 1))
                    dABBA_loss_real = CycleCond_Loss()(dA, dA_r)
                    # dABBA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)
                    if self.fp16:
                        self.dABBA_optimizerscaler.scale(dABBA_loss_real).backward()
                        self.dABBA_optimizerscaler.step(self.dABBA_optimizer)
                        self.dABBA_optimizerscaler.update()
                    else:
                        dABBA_loss_real.backward()
                        self.dABBA_optimizer.step()

                    self.dBAAB_optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        dB = self.d_BAAB(torch.cat([fake_A, imgs_B], 1))
                        dB_r = self.d_BAAB(torch.cat([fake_A, fake_BAAB], 1))
                    dBAAB_loss_real = CycleCond_Loss()(dB, dB_r)
                    # dABBA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)
                    if self.fp16:
                        self.dBAAB_optimizerscaler.scale(dBAAB_loss_real).backward()
                        self.dBAAB_optimizerscaler.step(self.dBAAB_optimizer)
                        self.dBAAB_optimizerscaler.update()
                    else:
                        dBAAB_loss_real.backward()
                        self.dBAAB_optimizer.step()

                    d_cycle_loss = 0.5 * (dBAAB_loss_real.item() + dABBA_loss_real.item())
                    b = datetime.datetime.now()
                    # print("Cycle discriminator train time", b - a)

                else:
                    d_cycle_loss = 0
                # Calculate the max margin loss for A->B, B->A
                ## Max margin AB and BA
                if self.one_way_mm:
                    self.g_AB_optimizer.zero_grad()
                    a = datetime.datetime.now()
                    with torch.cuda.amp.autocast():
                        mm_a = self.g_AB(imgs_A)
                    mm_a_loss = MaxMargin_Loss(batch_size=imgs_A.shape[0])(mm_a, imgs_B)

                    # Calling the step function on an Optimizer makes an update to its
                    # parameters
                    if self.fp16:
                        self.g_AB_optimizerscaler.scale(mm_a_loss).backward()
                        self.g_AB_optimizerscaler.step(self.g_AB_optimizer)
                        self.g_AB_optimizerscaler.update()
                    else:
                        mm_a_loss.backward(retain_graph=True)
                        self.g_AB_optimizer.step()
                    mm_a_loss = mm_a_loss.item()

                    self.g_BA_optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        mm_b = self.g_BA(imgs_B)
                    mm_b_loss = MaxMargin_Loss(batch_size=imgs_A.shape[0])(mm_b, imgs_A)
                    if self.fp16:
                        self.g_BA_optimizerscaler.scale(mm_b_loss).backward()
                        self.g_BA_optimizerscaler.step(self.g_BA_optimizer)
                        self.g_BA_optimizerscaler.update()
                    else:
                        mm_b_loss.backward()
                        self.g_BA_optimizer.step()
                    mm_b_loss = mm_b_loss.item()
                    b = datetime.datetime.now()
                    # print("MM one way discriminator train time", b - a)


                else:
                    mm_a_loss = mm_b_loss = 0
                # Calculate the cycle A->B->A, B->A->B with max margin, and mae
                a = datetime.datetime.now()
                self.combined_optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    fake_B = self.g_AB(imgs_A)
                    fake_A = self.g_BA(imgs_B)
                    # with torch.no_grad():
                    valid_A = self.d_A(fake_A)
                    valid_B = self.d_B(fake_B)
                valid_A_loss = nn.BCEWithLogitsLoss()(valid_A, valid)
                valid_B_loss = nn.BCEWithLogitsLoss()(valid_B, valid)
                id_a = fake_B
                id_b = fake_A
                if self.id_loss:
                    gamma = 1.0
                    mae_id_abba = gamma * torch.nn.L1Loss()(id_a, imgs_A)
                    mae_id_baab = gamma * torch.nn.L1Loss()(id_b, imgs_B)
                else:
                    mae_id_abba = mae_id_baab = 0
                with torch.cuda.amp.autocast():
                    fake_ABBA = self.g_BA(fake_B)
                    fake_BAAB = self.g_AB(fake_A)
                if self.cycle_mm:
                    mm_abba = MaxMargin_Loss(batch_size=imgs_A.shape[0])(fake_ABBA, imgs_A)
                    mm_baab = MaxMargin_Loss(batch_size=imgs_A.shape[0])(fake_BAAB, imgs_B)
                else:
                    mm_abba = mm_baab = 0

                if self.cycle_loss:
                    mae_abba = torch.nn.L1Loss()(fake_ABBA, imgs_A)
                    mae_baab = torch.nn.L1Loss()(fake_BAAB, imgs_B)
                else:
                    mae_abba = 0
                    mae_baab = 0
                if self.cycle_dis:
                    with torch.cuda.amp.autocast():
                        dA = self.d_ABBA(torch.cat([fake_B, imgs_A], 1))
                        dA_r = self.d_ABBA(torch.cat([fake_B, fake_ABBA], 1))
                        dABBA_loss_real = CycleCond_Loss()(dA, dA_r)
                        dB = self.d_BAAB(torch.cat([fake_A, imgs_B], 1))
                        dB_r = self.d_BAAB(torch.cat([fake_A, fake_BAAB], 1))
                    dBAAB_loss_real = CycleCond_Loss()(dB, dB_r)
                else:
                    dABBA_loss_real = 0
                    dBAAB_loss_real = 0
                g_loss = valid_A_loss + valid_B_loss + \
                         self.cycle_mm_weight * mm_abba + self.cycle_mm_weight * mm_baab + \
                         mae_abba + mae_baab + \
                         self.id_loss_weight * mae_id_abba + self.id_loss_weight * mae_id_baab + \
                         dBAAB_loss_real + dABBA_loss_real
                if self.fp16:
                    self.combined_optimizerscaler.scale(g_loss).backward()
                    self.combined_optimizerscaler.step(self.combined_optimizer)
                    self.combined_optimizerscaler.update()
                else:
                    g_loss.backward()
                    self.combined_optimizer.step()
                b = datetime.datetime.now()
                # print("Combined gen train time", b - a)

                if batch_i % 50 == 0 and batch_i != 0:
                    print(
                        "Epoch", epoch, "/", training_epochs,
                        "Batch:", batch_i, len(dataloader),
                        "Global Step", count,
                        "Discriminator loss:", d_loss,
                        # "Discriminator acc:", "{:.2f}".format(100 * np.mean(accs)),
                        "Combined loss:", "{:.2f}".format(g_loss.item()),
                        "MM_ABBA_CYCLE:", "{:.2f}".format(mm_abba.item() if self.cycle_mm else 0),
                        "MM_BAAB_CYCLE:", "{:.2f}".format(mm_baab.item() if self.cycle_mm else 0),
                        "abba acc:", "{:.2f}".format(mae_abba.item() if self.cycle_loss else 0),
                        "baab acc:", "{:.2f}".format(mae_baab.item() if self.cycle_loss else 0),
                        "idloss ab:", "{:.2f}".format(mae_id_abba.item() if self.id_loss else 0),
                        "idloss ba:", "{:.2f}".format(mae_id_baab.item() if self.id_loss else 0),
                        "mm ab loss:", "{:.2f}".format(mm_a_loss if self.one_way_mm else 0),
                        "mm ba loss:", "{:.2f}".format(mm_b_loss if self.one_way_mm else 0),
                        "discriminator cycle loss:", "{:.2f}".format(d_cycle_loss),
                    )
                    scalars = {
                        "epoch": epoch,
                        # "batch": batch_i,
                        "global_step": count,
                        "discriminator_loss": d_loss,
                        # "discriminator_acc": np.mean(accs),
                        "combined_loss": g_loss.item(),
                        "loss": g_loss.item() + d_loss,
                        "MM_ABBA_CYCLE": mm_abba.item() if self.cycle_mm else 0,
                        "MM_BAAB_CYCLE": mm_baab.item() if self.cycle_mm else 0,
                        "abba_mae": mae_abba.item() if self.cycle_loss else 0,
                        "baab_mae": mae_baab.item() if self.cycle_loss else 0,
                        "cycle_da": valid_A_loss.item(),
                        "cycle_db": valid_B_loss.item(),
                        "idloss_ab": mae_id_abba.item() if self.id_loss else 0,
                        "idloss_ba": mae_id_baab.item() if self.id_loss else 0,
                        "mm_ab_loss": mm_a_loss if self.one_way_mm else 0,
                        "mm_ba_loss": mm_b_loss if self.one_way_mm else 0,
                        "discriminator_cycle_loss": d_cycle_loss
                    }
                    if wdb:
                        wandb.log(scalars, step=count)
                    if tb:
                        writer.add_scalars("run", tag_scalar_dict=scalars, global_step=count)
                        writer.flush()

        def train_loop(training_epochs, iters=None):
            count = 0
            # We gave a specific amount of epochs
            if iters is None:
                for epoch in range(training_epochs):
                    for batch_i, (distributional_embeddings, retrofitted_embeddings) in enumerate(dataloader):
                        train_step(self, batch_i, distributional_embeddings, retrofitted_embeddings, epoch, count,
                                   training_epochs)
                        count += 1
                    print("\n")
                    sl, sv, c = self.test(dataset)
                    print(sl, sv, c)
                    print("Saving our results.")
                    # Save to tensorboard
                    if tb:
                        writer.add_scalar("simlex", sl, global_step=count)
                        writer.add_scalar("simverb", sv, global_step=count)
                        writer.add_scalar("card", c, global_step=count)
                        writer.flush()
                    # Save them also to wandb
                    if wdb:
                        wandb.log({"simlex": sl, "card": c, "simverb": sv, "epoch": epoch}, step=count)
                    if ray:
                        tune.report(**{"simlex": sl, "card": c, "simverb": sv, "epoch": epoch})
                    # Save a checkpoint
                    if epochs_per_checkpoint is not None:
                        if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                            self.save_model(name="checkpoint")
                    print("\n")
                    res.append((sl, sv, c))
                    print(res)
                    print("\n")
            else:
                epoch = 0
                running = True
                while running:
                    for batch_i, (distributional_embeddings, retrofitted_embeddings) in enumerate(dataloader):
                        if count >= iters:
                            running = False
                            break
                        train_step(self, batch_i, distributional_embeddings, retrofitted_embeddings, epoch, count,
                                   iters / len(dataloader))
                        count += 1
                    epoch += 1
                    print("\n")
                    sl, sv, c = self.test(dataset)
                    print(sl, sv, c)
                    # Save to tensorboard
                    if tb:
                        writer.add_scalar("simlex", sl, global_step=count)
                        writer.add_scalar("simverb", sv, global_step=count)
                        writer.add_scalar("card", c, global_step=count)
                        writer.flush()
                    # Save to wandb
                    if wdb:
                        wandb.log({"simlex": sl, "simverb": sv, "card": c}, step=count)
                    # Save the checkpoint
                    if epochs_per_checkpoint is not None:
                        if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                            self.save_model(name="checkpoint")
                    print('\n')
                    res.append((sl, sv, c))
                    print(res)
                    print("\n")

        # Start the training loop
        train_loop(epochs, iters=iters)
        print("Final performance")
        sl, sv, c = self.test(dataset)
        print(sl, sv, c)
        res.append((sl, sv, c))
        print('\n')
        return res

    def test(self, dataset,
             simlex="Data/testing/simlexorig999.txt",
             simverb="Data/testing/simverb3500.txt",
             card="Data/testing/card660.tsv",
             fast_text_location="Data/fasttext_model/cc.en.300.bin",
             prefix="en_"):
        '''Method to test out the model.'''
        simlex = os.path.join(self.local_dir,simlex)
        simverb = os.path.join(self.local_dir,simverb)
        card = os.path.join(self.local_dir,card)
        fast_text_location = os.path.join(self.local_dir,fast_text_location)
        self.to("cpu")
        self.set_fp32()
        self.put_eval()
        sl = helpertools.test_model(self.g_AB, dataset, dataset_location=simlex,
                                         prefix=prefix, pt=True)[0]
        sv = helpertools.test_model(self.g_AB, dataset, dataset_location=simverb,
                                         prefix=prefix, pt=True)[0]
        c = helpertools.test_model(self.g_AB, dataset, dataset_location=card,
                                        prefix=prefix, pt=True)[0]
        self.to(self.device)
        self.set_fp16()
        self.put_train()
        return sl, sv, c

    def to_device(self, device):
        self.device = device
        self.d_A=self.d_A.to(device)
        self.d_B=self.d_B.to(device)
        self.d_ABBA=self.d_ABBA.to(device)
        self.d_BAAB=self.d_BAAB.to(device)
        self.g_AB=self.g_AB.to(device)
        self.g_BA=self.g_BA.to(device)
        self.to(device)


    def save_model(self, name=""):
        try:
            print("Trying to save model...")
            os.makedirs(self.save_folder, exist_ok=True)
            torch.save(self, os.path.join(self.save_folder, name + "complete.bin"))
            print("Succeeded!")
        except Exception as e:
            print(e)

    @staticmethod
    def load_model(path, device="cpu"):
        try:
            print("Trying to load model...")
            return torch.load(path, map_location=device)
        except Exception as e:
            print(e)
            return None
