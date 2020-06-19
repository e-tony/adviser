import numpy as np


def divide_chunks(data, n):
    idxs = [i for i in range(len(data[0]))]
    for i in range(0, len(idxs), n):
        yield idxs[i : i + n]


def split_into_batches(data, batch_size):
    batches = list(divide_chunks(data, batch_size))

    batched_data = []

    for batch_idxs in batches:
        embs = []
        rels = []
        idxs = []

        for i in batch_idxs:
            embs.append(data[0][i])
            rels.append(data[1][i])
            idxs.append(data[2][i])

        batched_data.append((np.asarray(embs), np.asarray(rels), idxs))
    return batched_data
