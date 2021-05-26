import argparse
import math
import os
import random

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seen_words', type=str, default="oov_test_0_05/SimLex-999_cut_to_0_05.txt",
                        help='File that we will reduce')
    parser.add_argument('--all_constraints', type=str, default="synonyms.txt",
                        help='File that we will reduce')
    parser.add_argument('--output_dir',default="./")
    args = parser.parse_args()
    output_dir = args.output_dir
    seen_words = args.seen_words
    target_file = args.all_constraints
    split = target_file.split(".")
    split_2 = (seen_words.split("/")[-1]if "/" in seen_words else seen_words).split(".")[0].split("_")
    output_file = split[0]+"_reducedwith_"+split_2[0]+"_"+split_2[-2]+"_"+split_2[-1]+"."+split[1]
    print("Outputting to",output_dir,output_file)

    seen_words_list = []
    original_length = 0
    prefix = "en_"
    with open(seen_words) as f:
        for line in f:
            seen_words_list.append(line.replace(prefix,"").strip())
    lines_to_keep = []
    with open(target_file) as f:
        for line in tqdm(f):
            original_length += 1
            for word in seen_words_list:
                if word in line:
                    lines_to_keep.append(line)
                    break
    print("Keeping",len(lines_to_keep),"out of",original_length,"A percentage of",(len(lines_to_keep)*1.0)/(original_length))
    with open(os.path.join(output_dir,output_file),"w") as of:
        for tup in lines_to_keep:
            of.write(tup)
