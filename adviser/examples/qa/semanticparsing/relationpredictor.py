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
from neuralmodels.mlp import NN
from sklearn.metrics import f1_score, accuracy_score


class Trainer:
    def __init__(self, config, params):
        self.config = config
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NN(int(self.config["emb_dim"]), int(self.config["n_classes"]), n_hid1=int(self.config["h_dim1"]))
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
        self.model = torch.load(model_path)

    def get_ranks(self, outputs, y_true):
        ranks = []
        try:
            for i, instance in enumerate(outputs):
                idxs = {i:o for i,o in enumerate(instance.tolist())}
                sorted_idxs = {k: v for k, v in sorted(idxs.items(), key=lambda item: item[1], reverse=True)}
                rank = list(sorted_idxs.keys()).index(y_true[i])
                ranks.append(rank+1)
        except:
            print(y_true)
        return ranks

    def train(self, train_data, dev_data):
        epoch_loss = []
        epoch_train_metrics = []
        epoch_train_f1_metrics = []
        epoch_train_hits3 = []
        epoch_train_hits10 = []
        epoch_train_mrr = []
        epoch_dev_metrics = []
        epoch_dev_f1_metrics = []
        epoch_dev_report_metrics = []
        epoch_dev_hits3 = []
        epoch_dev_hits10 = []
        epoch_dev_mrr = []

        classes = list(range(281))

        batches = utils.split_into_batches(train_data, int(self.config["batch_size"]))

        for epoch in range(int(self.config["epochs"])):
            start_time = time.time()

            train_loss = 0
            train_metrics = []
            train_f1_metrics = []
            train_hits3 = []
            train_hits10 = []
            train_mrr = []
            dev_loss = 0
            dev_metrics = []
            dev_f1_metrics = []
            dev_hits3 = []
            dev_hits10 = []
            dev_mrr = []

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
                train_f1 = f1_score(
                            rels.tolist(),
                            outputs.argmax(1).tolist(),
                            average="macro",
                        )
                train_f1_metrics.append(train_f1)

                train_ranks = self.get_ranks(outputs, rels.tolist())
                _train_hits3 = metrics.hits_at_k(train_ranks, k=3)
                _train_hits10 = metrics.hits_at_k(train_ranks, k=10)
                train_hits3.append(_train_hits3)
                train_hits10.append(_train_hits10)

                _train_mrr = metrics.mrr(train_ranks)
                train_mrr.append(_train_mrr)

                if i % 500 == 0:
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

                        dev_ranks = self.get_ranks(dev_outputs, dev_rels.tolist())
                        _dev_hits3 = metrics.hits_at_k(dev_ranks, k=3)
                        _dev_hits10 = metrics.hits_at_k(dev_ranks, k=10)
                        dev_hits3.append(_dev_hits3)
                        dev_hits10.append(_dev_hits10)

                        _dev_mrr = metrics.mrr(dev_ranks)
                        dev_mrr.append(_dev_mrr)

                        if self.params.log_metrics:
                            mlflow.log_metric("Train loss", train_loss / len(train_metrics))
                            mlflow.log_metric("Train acc", train_acc)
                            mlflow.log_metric("Train hitsat3", _train_hits3)
                            mlflow.log_metric("Train hitsat10", _train_hits10)
                            mlflow.log_metric("Train mrr", _train_mrr)
                            mlflow.log_metric("Dev loss", dev_loss / len(dev_metrics))
                            mlflow.log_metric("Dev acc", dev_acc)
                            mlflow.log_metric("Dev f1", dev_f1)
                            mlflow.log_metric("Dev hitsat3", _dev_hits3)
                            mlflow.log_metric("Dev hitsat10", _dev_hits10)
                            mlflow.log_metric("Dev mrr", _dev_mrr)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            self.scheduler.step()
            self.scheduler.step()
            epoch_loss.append(train_loss / len(train_metrics))
            epoch_train_metrics.append(sum(train_metrics) / len(train_metrics))
            epoch_train_f1_metrics.append(sum(train_f1_metrics)/len(train_f1_metrics))
            epoch_train_hits3.append(sum(train_hits3)/len(train_hits3))
            epoch_train_hits10.append(sum(train_hits10)/len(train_hits10))
            epoch_train_mrr.append(sum(train_mrr)/len(train_mrr))
            epoch_dev_metrics.append(sum(dev_metrics) / len(dev_metrics))
            epoch_dev_f1_metrics.append(sum(dev_f1_metrics) / len(dev_f1_metrics))
            epoch_dev_hits3.append(sum(dev_hits3)/len(dev_hits3))
            epoch_dev_hits10.append(sum(dev_hits10)/len(dev_hits10))
            epoch_dev_mrr.append(sum(dev_mrr)/len(dev_mrr))

            print(
                "Epoch: %d" % (epoch + 1),
                " | time in %d minutes, %d seconds" % (mins, secs),
            )
            print(f"\tEpoch avg Loss: {sum(epoch_loss)/len(epoch_loss):.4f}(train)")
            print(
                f"\tEpoch avg Acc: {sum(epoch_train_metrics)/len(epoch_train_metrics):.4f} (train)"
            )
            print(
                f"\tEpoch avg F1: {sum(epoch_train_f1_metrics)/len(epoch_train_f1_metrics):.4f} (train)"
            )
            print(f"\tEpoch avg hitsat3: {sum(epoch_train_hits3)/len(epoch_train_hits3):.4f} (train)")
            print(f"\tEpoch avg hitsat10: {sum(epoch_train_hits10)/len(epoch_train_hits10):.4f} (train)")
            print(f"\tEpoch avg mrr: {sum(epoch_train_mrr)/len(epoch_train_mrr):.4f} (train)")
            
            print(
                f"\tEpoch avg Acc: {sum(epoch_dev_metrics)/len(epoch_dev_metrics):.4f} (dev)"
            )
            print(
                f"\tEpoch avg F1: {sum(epoch_dev_f1_metrics)/len(epoch_dev_f1_metrics):.4f} (dev)"
            )
            print(f"\tEpoch avg hitsat3: {sum(epoch_dev_hits3)/len(epoch_dev_hits3):.4f} (dev)")
            print(f"\tEpoch avg hitsat10: {sum(epoch_dev_hits10)/len(epoch_dev_hits10):.4f} (dev)")
            print(f"\tEpoch avg mrr: {sum(epoch_dev_mrr)/len(epoch_dev_mrr):.4f} (dev)")
            print(f"\tLast Acc: {epoch_dev_metrics[-1]:.4f} (dev)")
            print(f"\tLast F1: {epoch_dev_f1_metrics[-1]:.4f} (dev)")
            print(f"\tEpoch last hitsat3: {epoch_dev_hits3[-1]:.4f} (dev")
            print(f"\tEpoch last hitsat10: {epoch_dev_hits10[-1]:.4f} (dev")
            print(f"\tEpoch last mrr: {epoch_dev_mrr[-1]:.4f} (dev")

            if self.params.log_metrics:
                mlflow.log_metric("Epoch Loss", sum(epoch_loss) / len(epoch_loss))
                mlflow.log_metric(
                    "Epoch Avg Acc train",
                    sum(epoch_train_metrics) / len(epoch_train_metrics),
                )
                mlflow.log_metric(
                    "Epoch Avg F1 train",
                    sum(epoch_train_f1_metrics)/len(epoch_train_f1_metrics),
                )
                mlflow.log_metric("Epoch Hits at3 train", epoch_train_hits3[-1])
                mlflow.log_metric("Epoch Hits at10 train", epoch_train_hits10[-1])
                mlflow.log_metric("Epoch MRR train", epoch_train_mrr[-1])
                mlflow.log_metric(
                    "Epoch Avg Acc dev", sum(epoch_dev_metrics) / len(epoch_dev_metrics)
                )
                mlflow.log_metric(
                    "Epoch Avg F1 dev",
                    sum(epoch_dev_f1_metrics) / len(epoch_dev_f1_metrics),
                )
                mlflow.log_metric("Epoch Avg MRR dev", sum(epoch_dev_mrr)/len(epoch_dev_mrr))
                mlflow.log_metric("Epoch Acc dev", epoch_dev_metrics[-1])
                mlflow.log_metric("Epoch F1 dev", epoch_dev_f1_metrics[-1])
                mlflow.log_metric("Epoch Hits at3 dev", epoch_dev_hits3[-1])
                mlflow.log_metric("Epoch Hits at10 dev", epoch_dev_hits10[-1])
                mlflow.log_metric("Epoch MRR dev", epoch_dev_mrr[-1])

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
        self.save_model()


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
    parser.add_argument(
        "--config",
        required=True,
        default=False,
        help="Set to path of configuration file.",
    )
    parser.add_argument(
        "--random-emb",
        required=False,
        default=False,
        help="Set to true if sentence embeddings should be randomly generated.",
    )
    args = parser.parse_args()

    config_files = [os.path.join(args.config, f) for f in os.listdir(args.config)]

    for file in config_files:
        print("Working with configuration file:", file)

        config = configparser.ConfigParser()
        # config.read("default.conf")
        config.read(file)

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
            if args.random_emb:
                train_embs = np.random.uniform(-1, 1, (406213, 768))
                dev_embs = np.random.uniform(-1, 1, (45076, 768))
            else:
                train_embs = reader.load_embs(args.data_dir + "csqa.train.embeddings.bin")
                dev_embs = reader.load_embs(args.data_dir + "csqa.dev.embeddings.bin")
            train_data = reader.get_data(
                df_train,
                train_embs,
                rel2idx,
                subset=int(config["parameters"]["subset_train"]),
                shuffle=False,
                random=args.random_emb
            )

            # df_dev = pd.read_json(args.data_dir + "csqa.dev.json")
            # dev_embs = reader.load_embs(args.data_dir + "csqa.dev.embeddings.bin")
            dev_data = reader.get_data(
                df_dev,
                dev_embs,
                rel2idx,
                subset=int(config["parameters"]["subset_dev"]),
                shuffle=False,
                random=args.random_emb
            )
            trainer.train(train_data, dev_data)

        if args.log_metrics:
            mlflow.end_run()
