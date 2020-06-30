import os
import pandas as pd
import numpy as np
import json
import pickle
import math
import time
import configparser
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow

import utils
import reader
from evaluation import metrics
import neuralmodels
from neuralmodels.mlp import MLP, NN
from sklearn.metrics import f1_score, accuracy_score


class Trainer:
    def __init__(self, config, params):
        self.config = config
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NN(int(self.config["emb_dim"]), int(self.config["n_classes"]))
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=float(self.config["learning_rate"])
        )

        self.model_path = None
        if "model_path" in self.config:
            self.model_path = self.config["model_path"]
            self.load_model(self.model_path)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def train(self, train_data, dev_data):
        epoch_loss = []
        epoch_train_metrics = []
        epoch_dev_metrics = []
        epoch_dev_f1_metrics = []
        epoch_dev_report_metrics = []

        classes = list(range(281))

        batches = utils.split_into_batches(train_data, int(self.config["batch_size"]))

        for epoch in range(int(self.config["epochs"])):
            start_time = time.time()

            train_loss = 0
            train_metrics = []
            dev_loss = 0
            dev_metrics = []
            dev_f1_metrics = []

            for i, batch in enumerate(batches):
                self.optimizer.zero_grad()  # TODO what does it do?
                embs, rels, idxs = (
                    torch.tensor(batch[0]),
                    torch.tensor(batch[1], dtype=torch.long),
                    torch.tensor(batch[2]),
                )
                self.optimizer.zero_grad()
                embs, rels = embs.to(self.device), rels.to(self.device)
                outputs = self.model(embs)
                loss = F.cross_entropy(outputs, rels)
                # loss = self.criterion(outputs, rels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                #         train_loss += loss.detach()  # TODO why does this take so long?

                num_corrects = (outputs.argmax(1) == rels).sum().item()
                train_acc = num_corrects / len(rels)
                train_metrics.append(train_acc)

                if self.params.log_metrics:
                    mlflow.log_metric("Train loss", train_loss / len(train_metrics))
                    mlflow.log_metric("Train acc", train_acc)

                if i % 100 == 0:
                    with torch.no_grad():
                        dev_embs, dev_rels, dev_idxs = dev_data
                        dev_embs, dev_rels, dev_idxs = (
                            torch.tensor(dev_embs),
                            torch.tensor(dev_rels, dtype=torch.long),
                            torch.tensor(dev_idxs),
                        )
                        dev_embs, dev_rels = (
                            dev_embs.to(self.device),
                            dev_rels.to(self.device),
                        )
                        dev_outputs = self.model(dev_embs)
                        _loss = F.cross_entropy(dev_outputs, dev_rels)
                        dev_loss += _loss.item()
                        num_corrects = (dev_outputs.argmax(1) == dev_rels).sum().item()
                        dev_acc = num_corrects / len(dev_rels)
                        dev_metrics.append(dev_acc)
                        dev_f1 = f1_score(
                            dev_rels.tolist(),
                            dev_outputs.argmax(1).tolist(),
                            average="macro",
                        )
                        dev_f1_metrics.append(dev_f1)
                        dev_report = metrics.get_classification_report(
                            dev_rels.tolist(), dev_outputs.argmax(1).tolist(), classes
                        )
                        epoch_dev_report_metrics.append(dev_report)

                        if self.params.log_metrics:
                            mlflow.log_metric("Dev loss", dev_loss / len(dev_metrics))
                            mlflow.log_metric("Dev acc", dev_acc)
                            mlflow.log_metric("Dev f1", dev_f1)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            self.scheduler.step()
            self.scheduler.step()
            epoch_loss.append(train_loss / len(train_metrics))
            epoch_train_metrics.append(sum(train_metrics) / len(train_metrics))
            epoch_dev_metrics.append(sum(dev_metrics) / len(dev_metrics))
            epoch_dev_f1_metrics.append(sum(dev_f1_metrics) / len(dev_f1_metrics))

            print(
                "Epoch: %d" % (epoch + 1),
                " | time in %d minutes, %d seconds" % (mins, secs),
            )
            print(f"\tEpoch Loss: {sum(epoch_loss)/len(epoch_loss):.4f}(train)")
            print(
                f"\tEpoch Acc: {sum(epoch_train_metrics)/len(epoch_train_metrics):.4f} (train)"
            )
            print(
                f"\tEpoch Acc: {sum(epoch_dev_metrics)/len(epoch_dev_metrics):.4f} (dev)"
            )
            print(
                f"\tEpoch F1: {sum(epoch_dev_f1_metrics)/len(epoch_dev_f1_metrics):.4f} (dev)"
            )
            print(f"\tLast dev Acc: {epoch_dev_metrics[-1]:.4f} (dev)")
            print(f"\tLast dev F1: {epoch_dev_f1_metrics[-1]:.4f} (dev)")

            if self.params.log_metrics:
                mlflow.log_metric("Epoch Loss", sum(epoch_loss) / len(epoch_loss))
                mlflow.log_metric(
                    "Epoch Avg Acc train",
                    sum(epoch_train_metrics) / len(epoch_train_metrics),
                )
                mlflow.log_metric(
                    "Epoch Avg Acc dev", sum(epoch_dev_metrics) / len(epoch_dev_metrics)
                )
                mlflow.log_metric(
                    "Epoch Avg F1 dev",
                    sum(epoch_dev_f1_metrics) / len(epoch_dev_f1_metrics),
                )
                mlflow.log_metric("Epoch Acc dev", epoch_dev_metrics[-1])
                mlflow.log_metric("Epoch F1 dev", epoch_dev_f1_metrics[-1])

        # print final report
        print(epoch_dev_report_metrics[-1])

        if not os.path.isdir("saved_models"):
            os.mkdir("saved_models")
        if not self.model_path:
            self.model_path = "saved_models/" + utils.make_filename(self.config)
        else:
            self.model_path.split(".pt")[0] + "_" + time.strftime(
                "%Y%m%d-%H%M%S"
            ) + ".pt"
        torch.save(self.model.state_dict(), self.model_path)

    # def evaluate(self, dev_data):
    #     self.model.eval()
    #
    #     # record evaluation matrics
    #     total_loss = 0
    #     epoch_metrics = []
    #
    #     label_list = list(range(281))
    #     batches = utils.split_into_batches(
    #         dev_data, int(self.config["batch_size"])
    #     )  # eval in batches
    #
    #     predictions = []
    #     labels = []
    #
    #     with torch.no_grad():
    #         for batch in batches:
    #             embs, rels, idxs = (
    #                 torch.tensor(batch[0], device=self.device),
    #                 torch.tensor(batch[1], dtype=torch.long, device=self.device),
    #                 torch.tensor(batch[2], device=self.device),
    #             )
    #             outputs = self.model(embs)
    #             loss = F.cross_entropy(outputs, rels)
    #             total_loss += loss.item()
    #
    #             labels.append(rels.tolist())
    #             predictions.append(outputs.argmax(1).tolist())
    #
    #     # F1 metrics:
    #     # print("Labels", labels, "\n")
    #     # print("Predictions", predictions, "\n")
    #     # print("Label list:", label_list)
    #
    #     predictions = [pred for batch_preds in predictions for pred in batch_preds]
    #     labels = [label for batch_labels in labels for label in batch_labels]
    #     macrof1 = metrics.get_macro_f1(labels, predictions, label_list)
    #     microf1 = metrics.get_micro_f1(labels, predictions, label_list)
    #
    #     epoch_metrics.append({"macro-f1": macrof1, "micro-f1": microf1})
    #
    #     print("Macro F1:", macrof1)
    #     print("Micro F1:", microf1)
    #
    #     if self.params.log_metrics:
    #         mlflow.log_metric("Macro F1", macrof1)
    #         mlflow.log_metric("Micro F1", microf1)
    #
    #     return epoch_metrics

    # def test(self, data):
    #     assert self.model_path
    #     self.model.load_state_dict(torch.load(self.model_path))
    #     self.model.eval()
    #
    #     test_loss = 0
    #     test_metrics = {}
    #
    #     self.optimizer.zero_grad()
    #     embs, rels, idxs = (
    #         torch.tensor(data[0], device=self.device),
    #         torch.tensor(data[1], dtype=torch.long, device=self.device),
    #         torch.tensor(data[2], device=self.device),
    #     )
    #     outputs = self.model(embs)
    #
    #     loss = self.criterion(outputs, rels)
    #     test_loss += loss.item()
    #     #     metrics = get_metrics(output.argmax(10), rels)
    #     #     test_metrics = add_metrics(metrics, test_metrics)
    #
    #     return test_loss / len(data), test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
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
        "--log-metrics",
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
        mlflow.log_params(utils.get_log_params(config))

    trainer = Trainer(config["parameters"], args)
    if args.train:
        df_train = pd.read_json(args.data_dir + "csqa.train.json")
        df_dev = pd.read_json(args.data_dir + "csqa.dev.json")
        train_embs = reader.load_embs(args.data_dir + "csqa.train.embeddings.bin")
        dev_embs = reader.load_embs(args.data_dir + "csqa.dev.embeddings.bin")
        train_data = reader.get_data(
            df_train,
            train_embs,
            rel2idx,
            subset=int(config["parameters"]["subset_train"]),
            shuffle=False,
        )
        # dev_data = reader.get_data(df_dev)
        # trainer.train(train_data, dev_data)

        # if args.test:
        df_dev = pd.read_json(args.data_dir + "csqa.dev.json")
        dev_embs = reader.load_embs(args.data_dir + "csqa.dev.embeddings.bin")
        dev_data = reader.get_data(
            df_dev,
            dev_embs,
            rel2idx,
            subset=int(config["parameters"]["subset_dev"]),
            shuffle=False,
        )
        trainer.train(train_data, dev_data)

    print("Results")  # TODO pretty print to terminal

    if args.log_metrics:
        mlflow.end_run()
