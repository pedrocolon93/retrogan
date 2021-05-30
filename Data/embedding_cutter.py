from tqdm import tqdm

from Code.helpertools import load_embeddings

if __name__ == '__main__':
    vocab = []
    files = ["antonyms.txt", "synonyms.txt","simlexsimverb.words"]
    for f in files:
        with open(f,"r") as file:
            for line in file:
                s = line.split()
                for item in s:
                    item = item.replace("en_","").strip()
                    vocab.append(item)
    vocab = list(set(vocab))
    print("Here")
    path_to_embeddings = "glove.840B.300d.txt"
    amount = 3000000
    final_embeddings = []
    df = load_embeddings(path_to_embeddings,amount,return_dict=True)
    for word in tqdm(vocab):
        try:
            emb = df[word]
            emb = [str(i) for i in emb]
            final_embeddings.append(("en_"+word,emb))
        except:
            pass
    final_embeddings = [' '.join([x[0]]+x[1])for x in final_embeddings]
    path_to_cut_embeddings = "glove_unseen.txt"
    with open(path_to_cut_embeddings,"w") as out:
        for line in final_embeddings:
            out.write(line+"\n")
    print(final_embeddings)
