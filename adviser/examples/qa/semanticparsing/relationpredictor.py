import os
import pandas as pd
import numpy as np
import json
import pickle
import math
import configparser
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow

from semanticparsing import utils
from semanticparsing import reader
from semanticparsing.neuralmodels.mlp import MLP


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(self.config["emb_dim"], self.config["n_classes"])
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=self.config["learning_rate"]
        )

        self.model_path = None  # TODO
        pass

    def train(self, train_data, dev_data=None):
        epoch_loss = []
        epoch_metrics = []

        batches = utils.split_into_batches(train_data, self.config["batch_size"])

        for epoch in self.config["epochs"]:
            train_loss = 0
            train_metrics = {}
            for batch in batches:
                embs, rels, idxs = (
                    torch.tensor(batch[0], device=self.device),
                    torch.tensor(batch[1], dtype=torch.long, device=self.device),
                    torch.tensor(batch[2], device=self.device),
                )
                self.optimizer.zero_grad()
                outputs = self.model(embs)
                loss = self.criterion(outputs, rels)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            #             metrics = get_metrics(output.argmax(10), rel)
            #             train_metrics = add_metrics(metrics, train_metrics)

            self.scheduler.step()
            epoch_loss.append(train_loss / len(train_data))
            epoch_metrics.append(train_metrics)

            print("Epoch: %d" % (epoch + 1))
            print(f"\tLoss: {train_loss/len(train_data):.4f}(train)")

        # TODO Wen-Tseng
        # TODO evaluate on dev

        torch.save(self.model.state_dict(), self.model_path)

    def test(self, data):
        test_loss = 0
        test_metrics = {}

        self.optimizer.zero_grad()
        embs, rels, idxs = (
            torch.tensor(data[0], device=self.device),
            torch.tensor(data[1], dtype=torch.long, device=self.device),
            torch.tensor(data[2], device=self.device),
        )
        outputs = self.model(embs)

        loss = self.criterion(outputs, rels)
        test_loss += loss.item()
        #     metrics = get_metrics(output.argmax(10), rels)
        #     test_metrics = add_metrics(metrics, test_metrics)

        return test_loss / len(data), test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=False,
        default="/mount/studenten/dialog-system/2020/student_directories/Tsereteli_Tornike/data/",
        help="Set to path of data.",
    )
    parser.add_argument(
        "--train",
        required=False,
        default=False,
        help="Set to True if training, else set to False.",
    )
    parser.add_argument(
        "--test",
        required=False,
        default=False,
        help="Set to True if testing, else set to False.",
    )
    parser.add_argument(
        "--log_metrics",
        required=False,
        default=False,
        help="Set to True if metrics should me logged with mlflow, else set to False.",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("default.conf")

    # load tags
    tagset = pd.read_json(args.data_dir + "csqa_tags.json")

    idx2rel = tagset.to_dict()[0]
    rel2idx = {v: k for k, v in idx2rel.items()}

    if args.log_metrics:
        mlflow.start_run()
        mlflow.log_params(utils.get_log_params())  # TODO

    trainer = Trainer(config["parameters"])
    if args.train:
        df_train = pd.read_json(args.data_dir + "csqa.train.json")
        df_dev = pd.read_json(args.data_dir + "csqa.dev.json")
        train_embs = reader.load_embs(args.data_dir + "csqa.train.embeddings.bin")
        dev_embs = reader.load_embs(args.data_dir + "csqa.dev.embeddings.bin")
        train_data = reader.get_data(
            df_train, train_embs, rel2idx, subset=None, shuffle=False
        )
        trainer.train()

    if args.test:
        df_dev = pd.read_json(args.data_dir + "csqa.dev.json")
        dev_embs = reader.load_embs(args.data_dir + "csqa.dev.embeddings.bin")
        test_data = reader.get_data(
            df_dev, dev_embs, rel2idx, subset=None, shuffle=False
        )
        trainer.test()

    print("Results")  # TODO pretty print to terminal

    if args.log_metrics:
        mlflow.end_run()
