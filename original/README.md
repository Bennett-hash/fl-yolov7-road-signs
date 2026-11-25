# The 2023 notebooks

These are the Google Colab notebooks that produced the results in the paper,
unchanged, with the outputs they printed when they ran in August 2023. Only
their file names are new. They ran on a Tesla T4, against
[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) as it stood then
(commit `84932d7`), with Python 3.10 and PyTorch 2.0.1.
[`SHA256SUMS`](SHA256SUMS) identifies the files; `shasum -a 256 -c SHA256SUMS`
in this folder checks them. The download logs in the notebooks contain GitHub's
signed links to `yolov7-tiny.pt`, which expired in 2023.

| Notebook | Paper | Original file name |
| --- | --- | --- |
| [`exp1_centralised.ipynb`](notebooks/exp1_centralised.ipynb) | Experiment 1, Figure 2 | `Result_Normal/Sign_YoloV7_Con.ipynb` |
| [`exp2_16clients_17rounds_3epochs.ipynb`](notebooks/exp2_16clients_17rounds_3epochs.ipynb) | Experiment 2, Figure 3 | `Result_LSGD/Sign_YoloV7_LSGD(adjust).ipynb` |
| [`exp3_4clients_17rounds_3epochs.ipynb`](notebooks/exp3_4clients_17rounds_3epochs.ipynb) | Experiment 3, Figure 4 | `Result_LSGD_fewClient/Sign_YoloV7_LSGD(adjust).ipynb` |
| [`exp4_4clients_17rounds_10epochs.ipynb`](notebooks/exp4_4clients_17rounds_10epochs.ipynb) | Experiment 4, Figure 5 | `Result_LSGE_fewClient_MoreLocalEpoch/Sign_YoloV7_LSGD(adjust).ipynb` |
| [`exp5_4clients_40rounds_3epochs.ipynb`](notebooks/exp5_4clients_40rounds_3epochs.ipynb) | Experiment 5, Figure 6 | `Result_LSGD_fewClient_MoreGlobalEpoch/Sign_YoloV7_LSGD(adjust).ipynb` |

[`results/expN/`](results) holds what `test.py` saved when each notebook
scored its final model on the test set: the confusion matrix that is the
paper's figure, and the precision, recall, F1 and precision–recall curves.
`results/exp1/train/` adds the centralised run's training log. The pictures of
sample predictions that `test.py` also saved are left out, because they show
GTSRB images. The final models of the five runs are in the repository's
[release](https://github.com/Bennett-hash/fl-yolov7-road-signs/releases).

In these confusion matrices, as YOLOv7 drew them at this commit, the cell for
a sign of class A detected as class B is in row A, column B: the other way
round from the axis labels "Predicted" (rows) and "True" (columns), which fit
only the last row and column.

## What they report

The last cell of each notebook scores the final model on the 3,921 test
images. These are the numbers it printed:

| Experiment | Clients | Rounds | Local epochs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --: | --: | --: | --: | --: | --: | --: |
| 1 | – | – | 50 | 0.495 | 0.779 | 0.646 | 0.430 |
| 2 | 16 | 17 | 3 | 0.320 | 0.273 | 0.065 | 0.037 |
| 3 | 4 | 17 | 3 | 0.428 | 0.399 | 0.216 | 0.135 |
| 4 | 4 | 17 | 10 | 0.479 | 0.695 | 0.584 | 0.370 |
| 5 | 4 | 40 | 3 | 0.493 | 0.424 | 0.343 | 0.195 |

The paper quotes 0.642 for Experiment 1, which is the validation score
`train.py` printed at the end of training; the test score above is 0.646. For
Experiment 2 it says "less than 0.1", and for Experiments 3 to 5 it gives no
score.

## The averaging error in Experiments 2 to 5

Cell 10 of notebooks 2 to 5 defines `average_weights`, which was meant to
average the clients' models. Its key line is

```python
model1.state_dict()[name] = params
```

`state_dict()` builds a new dictionary each time it is called, so this stores
the average in a dictionary that is then thrown away. The model is never
changed, and the function returns the first client's model as it was. Every
round therefore carried on from client 1 alone, and the other clients'
training had no effect.

So Experiments 2 to 5 did not measure federated averaging. Each measured
training on one client's share of the data (1/16 of it in Experiment 2, 1/4 in
the others), with `train.py` restarted from that client's own previous weights
every round, that is every 3 or 10 epochs.

The function is kept, word for word, in
[`src/fedyolo/original.py`](../src/fedyolo/original.py), and
[`tests/test_averaging.py`](../tests/test_averaging.py) shows that it returns
the first client unchanged. The
[repository's README](../README.md#results) gives the results of the re-runs.

## The dataset they used

The notebooks unpacked a dataset that was not prepared as Section 2.1 of the
paper describes: each image was the GTSRB image cropped to its region of
interest, and each box was the region of interest of a different image of the
same class. This affects all five experiments. Its split also puts 3 images of
every GTSRB track, and so of every physical sign, in the test set, 3 in the
validation set and the rest in the training set.
[Notes on the data](../README.md#notes-on-the-data) gives the details.

## Other differences from the paper's text

- The paper says GTSRB has "over 50,000 images distributed across 40
  classes". The notebooks use its 43 classes and the 39,209 images of its
  training set, divided 31,367 / 3,921 / 3,921.
- The paper calls the image resolution used "the theoretical median of the
  image resolution of the training set". The notebooks use 64 × 64 pixels; the
  median image of the training set is 43 × 43.
- The paper gives "a consistent learning rate of 0.001". The runs used
  YOLOv7's default schedule (`data/hyp.scratch.p5.yaml`), which starts again
  with every call of `train.py`: a warm-up of three epochs or 1,000
  iterations, whichever is longer, and a cosine fall from 0.01 towards 0.001,
  which it comes close to only at the end of a long run such as the 50-epoch
  centralised one.
- The paper says Experiment 4 "outperformed the centralized approach". By the
  scores above it did not: 0.584 against 0.646 mAP@0.5.

## Why the notebooks are unchanged

They are the record of what was run. The corrected code is a separate,
tested package, so that anyone can compare the two.
