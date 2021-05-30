from tqdm import tqdm

from Code.helpertools import load_embeddings

if __name__ == '__main__':
    slsv_words = []
    with open("simlexsimverb.words","r") as f:
        for line in f:
            slsv_words.append(line.strip())
    print("Cleaning ants")
    with open("antonyms.txt","r") as ants:
        with open("antonyms_clean.txt","w")  as ants_out:
            for line in tqdm(ants):
                s = line.split()
                for word in s:
                    word = word.strip()
                    if word in slsv_words:
                        # print("Skipping",word)
                        continue
                    else:
                        ants_out.write(word)
    print("Cleaning syns")
    with open("synonyms.txt","r") as ants:
        with open("synonyms_clean.txt","w")  as ants_out:
            for line in tqdm(ants):
                s = line.split()
                for word in s:
                    word = word.strip()
                    if word in slsv_words:
                        # print("Skipping", word)
                        continue
                    else:
                        ants_out.write(word)
    print("Done")
