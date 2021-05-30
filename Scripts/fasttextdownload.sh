#!/usr/bin/env bash
cd ../Data

echo "Getting text";
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
echo "Extracting text";
gunzip cc.en.300.vec.gz;
rm cc.en.300.vec.gz;
echo "Done\nGetting Binary";
mkdir fasttext_model
cd fasttext_model
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz;
echo "Extracting"
gunzip cc.en.300.bin.gz;
echo "Done"
