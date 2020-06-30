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


def make_filename(config):
    filename = ""
    for i, para in enumerate(config):
        sep = "&" if i > 0 else ""
        if "model_path" in para:
            continue
        # elif "stopwords_path" in para:
        #     stopwords = config[para].split("/")[-1].split(".")[0]
        #     filename += f"{stopwords}-"
        #     continue
        filename += f"{sep}{para}={config[para]}"
    return filename + ".pt"


def get_log_params(config):
    params = {}
    for sec in config:
        for k, v in config[sec].items():
            params[sec + "/" + k] = v
    return params
