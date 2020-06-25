import metrics

def evaluate(self, dev_data):
    epoch_loss = []
    epoch_metrics = []

    # record evaluation matrics
    best_dev_loss = float('inf')
    best_predictions = []
    true_labels = []
    epoch_macrof1 = []
    epoch_microf1 = []

    self.model.eval()

    batches = utils.split_into_batches(dev_data, self.config["batch_size"])

    for epoch in self.config["epochs"]:
        dev_loss = 0
        dev_metrics = {}

        # record evaluation matrics
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in batches:
                embs, rels, idxs = (
                    torch.tensor(batch[0], device=self.device),
                    torch.tensor(batch[1], dtype=torch.long, device=self.device),
                    torch.tensor(batch[2], device=self.device),
                )
                outputs = self.model(embs) # predictions
                loss = self.criterion(outputs, rels)
                dev_loss += loss.item()

                # record for evaluation metrics:
                labels.append(rels)
                predictions.append(outputs)

            epoch_loss.append(dev_loss / len(dev_data))
            epoch_metrics.append(dev_metrics)

            # record for evaluation metrics:
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_predictions = predictions
                true_labels = labels

            # F1 metrics:
            macrof1 = get_macro_f1(labels, predictions)
            microf1 = get_micro_f1(labels, predictions)
            epoch_macrof1.append(macrof1)
            epoch_microf1.append(microf1)

            print("Epoch: %d" % (epoch + 1))
            print(f"\tLoss: {dev_loss/len(dev_data):.4f}(dev)")



    # find model with best dev loss
    # print its report table
    best_dev_loss = min(epoch_loss)
    print("Best dev loss: ", best_dev_loss)
    print_classification_report(true_labels, best_predictions, list(range(281)))
