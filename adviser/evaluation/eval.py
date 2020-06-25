def evaluate(self, dev_data):
    epoch_loss = []
    epoch_metrics = []

    self.model.eval()

    batches = utils.split_into_batches(dev_data, self.config["batch_size"])

    for epoch in self.config["epochs"]:
        dev_loss = 0
        dev_metrics = {}
        # add f1 metrix
        # add true_labels = []

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

            epoch_loss.append(dev_loss / len(dev_data))
            epoch_metrics.append(dev_metrics)

            print("Epoch: %d" % (epoch + 1))
            print(f"\tLoss: {dev_loss/len(dev_data):.4f}(dev)")

    # find model with best dev loss
    best_dev_loss = min(epoch_loss)
    print("Best_dev_loss: ", best_dev_loss)
