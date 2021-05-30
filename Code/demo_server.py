import numpy
import pandas as pd
import torch
from flask import Flask, request, jsonify
import faiss
import numpy as np
from flask_cors import CORS
from tqdm import tqdm

from helpertools import load_embeddings

app = Flask(__name__)
CORS(app)

def load_retrogan(path_to_retrogan):
    return torch.load(path_to_retrogan,map_location="cpu")


def post_specialize_embeddings(distributional_embeddings, retrogan):
    dist = torch.tensor(distributional_embeddings.values,dtype=torch.float32)
    bs = 128
    res = []
    with torch.no_grad():
        for i in tqdm(range(0,dist.shape[0],bs)):
            ps = retrogan(dist[i:i+bs])
            res.append(ps)
    ps_fin = torch.cat(res,0)
    ps_fin = pd.DataFrame(data=ps_fin.numpy(),index=distributional_embeddings.index)
    return ps_fin


def find_neighbors(term, embeddings,k=10):
    query = np.array([embeddings.loc[term]],dtype=numpy.float32)
    arr = np.array(embeddings.values,dtype=numpy.float32)
    xb = np.ascontiguousarray(arr)
    # make faiss available
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(xb)  # add vectors to the index
    D, I = index.search(query, k)  # sanity check
    items = [idx for idx in embeddings.iloc[np.squeeze(I)].index]
    distances = []
    for x in I[0]:
        a=torch.tensor(query)
        b=torch.tensor(embeddings.iloc[x]).unsqueeze(0)
        distances.append(torch.cosine_similarity(a,b).item())
    return items, distances

@app.route('/find_neighbors',methods=["POST"])
def hello_world():
    global distributional_embeddings, ps_embeddings
    input_dictionary = request.get_json()
    term = input_dictionary["input"]
    amount = int(input_dictionary["amount"])
    regular_neighbors = find_neighbors(term,distributional_embeddings,amount)
    ps_neighbors = find_neighbors(term,ps_embeddings,amount)
    r = jsonify({
        "regular":regular_neighbors,
        "post_specialized":ps_neighbors
    })
    return r

if __name__ == '__main__':
    path_to_embeddings = "../Data/cc.en.300.cut.vec"
    path_to_retrogan = "../Data/trained_retrogans/full_nb/checkpointcomplete.bin"
    emb_amount_to_load = 200000
    global distributional_embeddings, ps_embeddings
    distributional_embeddings = load_embeddings(path_to_embeddings, emb_amount_to_load)
    retrogan = load_retrogan(path_to_retrogan)
    ps_embeddings = post_specialize_embeddings(distributional_embeddings,retrogan)
    app.run(host="0.0.0.0",port="4000")