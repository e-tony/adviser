import numpy as np
import pickle


def load_embs(file):
    embs = []

    with open(file, "rb") as f:
        embs = pickle.load(f)

    return embs


def get_data(df, embeddings, rel2idx, subset=None, shuffle=False):
    assert len(df.tokens) == len(embeddings)
    assert len(df.tokens) == len(df.relation)
    data = []

    sub = len(df.relation)
    if subset:
        sub = subset

    if shuffle:
        pass

    embs = []
    rels = []
    idxs = []

    for i in range(len(df.relation))[:sub]:
        cls_emb = embeddings[i][0]
        #         data.append((np.asarray(cls_emb, dtype=np.float32), rel2idx[df.relation[i]], df.id[i]))
        embs.append(np.asarray(cls_emb, dtype=np.float32))
        rels.append(rel2idx[df.relation[i]])
        idxs.append(df.id[i])

    #     return data  # (emb (768,), rel, id)
    return (
        np.asarray(embs),
        np.asarray(rels, dtype=np.int16),
        np.asarray(idxs, dtype=np.int32),
    )
