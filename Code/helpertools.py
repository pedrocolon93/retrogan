import csv
import os
import fasttext
import numpy as np
import pandas as pd
import torch
from conceptnet5.vectors import cosine_similarity
from scipy.stats import pearsonr, spearmanr


def load_all_words_dataset_final(original, retrofitted, save_folder="./", cache=True, return_idx=False):
    '''Loads the word embeddings found in original/retrofitted and aligns them'''
    if cache:
        # If cache, simply load it up.
        print("Reusing cache")
        X_train = pd.read_hdf(os.path.join(save_folder, "filtered_x"), 'mat', encoding='utf-8')
        Y_train = pd.read_hdf(os.path.join(save_folder, "filtered_y"), 'mat', encoding='utf-8')
        if not return_idx:
            return np.array(X_train.values), np.array(Y_train.values)
        else:
            return np.array(X_train.values), np.array(Y_train.values), np.array(X_train.index)
    # else, load the files and intersect to find the ones that are in both datasets.
    print("Searching")
    print("for:", original, retrofitted)
    if "hdf" in original or "h5" in original:
        print("Assuming its a hdf file")
        o = pd.read_hdf(original, 'mat', encoding='utf-8')
        r = pd.read_hdf(retrofitted, 'mat', encoding='utf-8')
    else:
        print("Assuming its a text file")
        o = load_text_embeddings(original)
        r = load_text_embeddings(retrofitted)
    cns = r.index.intersection(o.index)
    print("Intersecting on", len(cns)) # This is done so that we can have distributional vec -> retrofitted
    # counterpart pairs
    X_train = o.loc[cns, :]
    Y_train = r.loc[cns, :]
    print('The final amount of vectors are', X_train.shape)
    if cache:
        # If cache, save the cache
        print("Dumping training")
        X_train.to_hdf(os.path.join(save_folder, "filtered_x"), "mat")
        Y_train.to_hdf(os.path.join(save_folder, "filtered_y"), "mat")

    print("Returning")
    return X_train, Y_train


def test_model(model, dataset, dataset_location='SimLex-999.txt',
             fast_text_location="../Data/fasttext_model/cc.en.300.bin",prefix="",pt=False,use_ft=False):
    word_tuples = []
    my_word_tuples = []
    global ft_model
    ds_model = None
    # if ft_model is None:
    #     ft_model= fasttext.load_model(fast_text_location)
    if isinstance(dataset,pd.DataFrame):
        ds_model = dataset
    elif ".txt" in dataset["original"]:
        ds_model = load_text_embeddings(dataset["original"])

    elif dataset is not None:
        ds_model = pd.read_hdf(dataset["original"],"mat")
        # ds_model=ds_model.swapaxes(0,1)
    retrogan = model
    with open(dataset_location) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        if use_ft:
            global ft_model
            if ft_model is None:
                ft_model = fasttext.load_model(fast_text_location)
        for row in csv_file:
            # print(f'Word1:\t{row[0]}\tWord2:\t{row[1]}\tSimscore:\t{row[2]}.')
            row = row.split()
            line_count += 1
            wtrow = []
            wtrow.append(row[0])
            wtrow.append(row[1])
            try:
                wtrow.append(row[3])
            except:
                wtrow.append(row[2])
            word_tuples.append(wtrow)
            score = 0
            try:
                if use_ft:
                    mw1 = ft_model.get_word_vector(row[0])
                    mw2 = ft_model.get_word_vector(row[1])
                else:
                    mw1 = ds_model.loc[prefix+row[0],:]
                    mw2 = ds_model.loc[prefix+row[1],:]
                mw1 /= np.linalg.norm(mw1)
                mw2 /= np.linalg.norm(mw2)
                if pt:
                    mw1 = np.array(retrogan(torch.tensor(np.array(mw1).reshape(1, 300),dtype=torch.float32)).detach()).reshape((300,))
                else:
                    mw1 = np.array(retrogan.predict(np.array(mw1).reshape(1, 300))).reshape((300,))
                # mw2 = ft_model.get_word_vector(row[1].lower())
                if pt:
                    mw2 = np.array(retrogan(torch.tensor(np.array(mw2).reshape(1, 300),dtype=torch.float32)).detach()).reshape((300,))
                else:
                    mw2 = np.array(retrogan.predict(np.array(mw2).reshape(1, 300))).reshape((300,))
                mw1 /= np.linalg.norm(mw1)
                mw2 /= np.linalg.norm(mw2)

                score = np.inner(mw1, mw2)
                del mw1, mw2
            except Exception as e:
                print(e)
                score = [0]
            my_word_tuples.append((row[0], row[1], score))

        # print(f'Processed {line_count} lines.')
    # pr = pearsonr([float(x[2]) for x in word_tuples], [float(x[2]) for x in my_word_tuples])
    # print(pr)
    sr = spearmanr([x[2] for x in word_tuples], [x[2] for x in my_word_tuples])
    # print(sr)
    # word_tuples = sorted(word_tuples,key=lambda x:(x[0],x[2]))
    return sr

def test_original_vectors(dataset, dataset_location='SimLex-999.txt', prefix=""):
    '''Similar to test_model, only it tests on the dataset, this is used to calculate distributional scores'''
    word_tuples = []
    my_word_tuples = []
    ds_model = None
    if isinstance(dataset, pd.DataFrame):
        ds_model = dataset
    elif dataset is not None:
        ds_model = pd.read_hdf(dataset["original"], "mat")

    # ds_model=ds_model.swapaxes(0,1)
    with open(dataset_location) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        missing = 0

        for row in csv_file:
            row = row.split()
            # print(f'Word1:\t{row[0]}\tWord2:\t{row[1]}\tSimscore:\t{row[2]}.')
            line_count += 1
            wtrow = []
            wtrow.append(row[0])
            wtrow.append(row[1])
            try:
                wtrow.append(row[3])
            except:
                wtrow.append(row[2])
            word_tuples.append(wtrow)
            score = 0
            try:
                mw1 = np.array(ds_model.loc[prefix + row[0], :])
                mw2 = np.array(ds_model.loc[prefix + row[1], :])
                mw1 /= np.linalg.norm(mw1)
                mw2 /= np.linalg.norm(mw2)
                score = np.inner(mw1, mw2)
            except Exception as e:
                # print(e)
                missing += 1
                score = 0
            my_word_tuples.append((row[0], row[1], score))
        # Calculate scores
        pr = pearsonr([float(x[2]) for x in word_tuples], [float(x[2]) for x in my_word_tuples])
        print("Pearson",pr)
        sr = spearmanr([x[2] for x in word_tuples], [x[2] for x in my_word_tuples])
        print("Spearman",sr)
        # word_tuples = sorted(word_tuples,key=lambda x:(x[0],x[2]))
        print("Missing", missing)
        return sr[0]


def load_text_embeddings(original):
    '''Loads a textual vector dataset into a dataframe.'''
    vecs = []
    idxs = []
    with open(original) as f:
        for line in f:
            line = line.strip()
            ls = line.split()
            if line != "" and len(ls) > 2:
                vecs.append([float(x) for x in ls[1:]])
                idxs.append(ls[0])
    return pd.DataFrame(index=idxs, data=vecs)
