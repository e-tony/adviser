# Evaluation matrix

The relation prediction task is a multi-class classifier.

First we calculate precision and recall for every class to get the F1 score.

`F1 score`: 2 * precision * recall / (precision + recall)

* F1 score is usually used in binary classification.

* We use F1 score to calculate macro F1.

`Macro F1`: sum(per-class F1) / number of class

* Macro F1 is simply the harmonic mean and treats every category equally.

`Micro F1`: 2 * micro precision * micro recall / (micro precision + micro recall)

* micro precision = sum(TP) / sum(TP + FP)

* micro recall = sum(TP) / sum(TP + FN)

* Micro F1 is usually used in unbalanced distributed dataset, i.e. the sample space of each category is extremely different.

* If dominate class has better predictions, the micro average will increase (bigger than macro).

`Mean Reciprocal Rank (MRR)`: sum(1/rank) / number of class. (if q retrieve no relevant document, reciprocal rank = 0)

* The mean of all reciprocal ranks for the true candidates over the test set (1/rank). (NLP-progress.com)

* The result of MRR is between 0 and 1, and 1 means perfect retrieval. 

`Hits at k (H@k)`:

* The rate of correct entities appearing in the top k entries for each instance list. This number may exceed 1.00 if the average k-truncated list contains more than one true entity. (NLP-progress.com)

..TBC
