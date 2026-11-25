# Correction to "Empirical validation of federated learning with YOLO v7-tiny for road sign detection: A simulation-based comparative study"

Yizhen Bi<br>
Department of Electrical and Electronic Engineering, The University of
Manchester, Manchester, M13 9PL, UK<br>
ORCID [0009-0005-5639-5771](https://orcid.org/0009-0005-5639-5771)

**Article corrected.** Yizhen Bi. Empirical validation of federated learning
with YOLO v7-tiny for road sign detection: A simulation-based comparative
study. *Applied and Computational Engineering* 49, 92–101, 2024. Proceedings
of the 4th International Conference on Signal Processing and Machine Learning
(CONF-SPML 2024).
[doi:10.54254/2755-2721/49/20241067](https://doi.org/10.54254/2755-2721/49/20241067).
Also recorded at CNKI as
[doi:10.26914/c.cnkihy.2024.074802](https://doi.org/10.26914/c.cnkihy.2024.074802).

**Status.** Written in November 2026 for the publisher, who has not yet
published a correction.

## Summary

In November 2026, while preparing the article's code for public release, I
found two errors in its experiments. In Experiments 2 to 5 the clients' models
were never averaged: every round continued from the first client's model, so
these experiments measured one client training alone. The dataset was also not
prepared as Section 2.1 describes: each image had been cropped to its region
of interest, and each bounding box came from a different image of the same
class. Five statements in the text are also incorrect. I have corrected both
errors and run all five experiments again.

All scores and Figures 2 to 6 change. On the test set the centralised model
scores 0.363 mAP@0.5 and the best federated setting 0.389; the 2023 runs gave
0.646 and 0.584. The article's statement that federated learning outperforms
centralised training under specific conditions does not hold: at its best
setting it reached the accuracy of centralised training, with 3.4 times the
computation. Its finding that the accuracy of federated learning depends
strongly on the number of clients, local epochs and rounds (the article's
global iterations) holds.

In all experiments, those of 2023 and the corrected ones, the test images are
other images of the signs in the training set (Section 1.3).

## 1. The errors

### 1.1 The clients' models were not averaged (Experiments 2 to 5)

Section 2.2 says that after the clients have trained, "an averaging mechanism
is invoked to aggregate these localisation weights into a uniform global model
weight", through "a custom function that calculates the arithmetic mean of all
subset weights". The function did not do this. It computed the mean but stored
it in a dictionary that was then discarded, and returned the first client's
model unchanged.

In Experiments 2 to 5 every round therefore continued from the first client's
model alone, and the other clients' training had no effect. The results given
for these experiments, in Figures 3 to 6 and Sections 3.2 to 3.5, are those of
one client training on its own share of the data (one sixteenth in Experiment
2, one quarter in Experiments 3 to 5), not of federated averaging.

### 1.2 The dataset was not prepared as Section 2.1 describes (all experiments)

Section 2.1 says that each GTSRB image was converted to PNG and labelled with
its own region of interest, normalised by the image's size (Equation 1). I had
made the dataset the experiments used differently, with a conversion script
that I did not keep. Comparing it with the official GTSRB files image by image
shows that:

- every image was the GTSRB image cropped to its region of interest, not the
  whole image; and
- every label had the right class, but its box was the region of interest of
  another image of the same class, normalised by that other image's size. Only
  78 of the 39,209 labels carried the box of the image they were cropped from,
  and since the images were cropped, even these boxes did not fit.

The boxes therefore covered about three quarters of the width and height of
images that were cropped to the sign. All five experiments used this dataset.

### 1.3 The test set shares signs with the training set (all experiments)

Section 2.1 says that the dataset was divided 8:1:1. Within each class, in
GTSRB's order, every tenth image was taken for validation and another tenth
for test. GTSRB's training set consists of 1,307 tracks, each of 30
consecutive images of one physical sign (one track has 29). Every track
therefore has 3 images in the validation set, 3 in the test set and the rest
in the training set. The test scores, those of 2023 and the corrected ones,
measure the detection of signs that are also in the training set, not of new
signs. The corrected experiments keep this division, so that they can be
compared with the 2023 runs.

### 1.4 Incorrect statements in the text and figures

- **Section 2.1** says GTSRB contains "over 50,000 images distributed across
  40 classes". The experiments used the 39,209 images of GTSRB's training
  set, in its 43 classes, divided into 31,367 training, 3,921 validation and
  3,921 test images.
- **Section 2.2** calls the image resolution used "the theoretical median of
  the image resolution of the training set". The resolution used was 64 × 64
  pixels; the median image of the training set is 43 × 43.
- **Section 2.3** gives "a consistent learning rate of 0.001". The runs used
  YOLO v7's default schedule, which starts again with every training run: a
  warm-up of three epochs or 1,000 iterations, whichever is longer, and a
  cosine fall from 0.01 towards 0.001 over the run's epochs.
- **Section 3.1** gives the centralised model's mAP as 0.642. That is its
  score on the validation set at the end of training. Its score on the test
  set, on which the confusion matrices and the other experiments were
  evaluated, was 0.646.
- **The abstract and Section 3.4** say that federated learning outperformed
  the centralised approach (Section 3.4: in Experiment 4). The 2023 runs did
  not show this: Experiment 4 scored 0.584 on the test set and the centralised
  model 0.646.
- **Figures 2 to 6** were drawn by YOLO v7. At the version used, it holds the
  cells for a sign of one class detected as another the other way round from
  the figure's axis labels. The corrected figures have the predicted class in
  rows and the true class in columns in every cell.

## 2. The corrected experiments

All five experiments of Table 2 were run again in November 2026 with both
errors corrected:

- **Dataset.** Built from the official GTSRB training archive as Section 2.1
  describes: each image converted to PNG and labelled with its own region of
  interest by Equation 1. The division into training, validation and test
  images is the one used in 2023, image for image.
- **Averaging.** The global model is the arithmetic mean of the clients'
  models over every floating-point tensor, the BatchNorm statistics included.
  The clients hold equal shares of the data, so this is also the
  sample-weighted mean of federated averaging.
- **Everything else as in 2023.** YOLO v7-tiny from its pretrained weights,
  YOLO v7 at commit 84932d7, the version used in 2023, images of 64 × 64, a
  batch size of 16, YOLO v7's default hyperparameters, and the clients, rounds
  (the article's global iterations) and local epochs of Table 2. As in 2023,
  each client returns the local epoch that scored highest on the validation
  set, which all clients share. The training images were divided among the
  clients at random with a fixed seed; in 2023 they were divided in the order
  the file system listed them, which was not recorded.
- **Environment.** Google Colab, NVIDIA L4 and Tesla T4 GPUs, Python 3.13 and
  PyTorch 2.14.

The five experiments were also run on the 2023 dataset with only the averaging
corrected.

Each experiment was run once, except the centralised one. Its first run on
each dataset was stopped by Colab and resumed (twice on the dataset as
described, four times on the 2023 dataset), so I ran it again from start to
finish. These repeats were started after the interrupted runs had ended and
their scores were known; I decided to report the repeats before they ended.
The interrupted runs scored 0.381 on the dataset as described and 0.531 on the
2023 dataset.

## 3. Corrected results

Table A replaces the results given in Sections 3.1 to 3.5. Scores are on the
3,921 test images, for each experiment's final model.

**Table A.** Corrected test results (dataset as described in Section 2.1,
averaging corrected).

<!-- correction-table-a:start -->
| Experiment | Setting | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --: | --: | --: | --: |
| 1 | Centralised, 50 epochs | 0.676 | 0.455 | 0.363 | 0.291 |
| 2 | 16 clients, 17 rounds, 3 local epochs | 0.507 | 0.306 | 0.068 | 0.049 |
| 3 | 4 clients, 17 rounds, 3 local epochs | 0.409 | 0.488 | 0.179 | 0.140 |
| 4 | 4 clients, 17 rounds, 10 local epochs | 0.597 | 0.476 | 0.389 | 0.323 |
| 5 | 4 clients, 40 rounds, 3 local epochs | 0.378 | 0.539 | 0.338 | 0.276 |
<!-- correction-table-a:end -->

**Table B.** Test mAP@0.5 of the 2023 runs, and with each error corrected in turn.

<!-- correction-table-b:start -->
| Experiment | 2023 runs | Averaging corrected, 2023 dataset | Averaging and dataset corrected |
| --- | --: | --: | --: |
| 1 | 0.646 | 0.577 | 0.363 |
| 2 | 0.065 | 0.073 | 0.068 |
| 3 | 0.216 | 0.238 | 0.179 |
| 4 | 0.584 | 0.547 | 0.389 |
| 5 | 0.343 | 0.383 | 0.338 |
<!-- correction-table-b:end -->

"2023 runs" gives the test scores the 2023 notebooks printed. The article
itself gives only 0.642 for Experiment 1, a validation score, and "less than
0.1" for Experiment 2.

Experiment 1 involves no averaging. Its middle column is the 2023 experiment
run again in the 2026 environment, with the same data and training arguments.
It scored 0.577 against 0.646 in 2023. The two runs differ in software
versions, in the number of data-loading processes and in sharing a GPU; which
of these caused the difference was not established.

All scores are lower on the dataset as described than on the 2023 dataset. In
the same environment, centralised training reaches 0.363 on the first and
0.577 on the second. The boxes are distributed alike in the two datasets:
centred, and about three quarters of the image wide and high. In the 2023
dataset the image is the sign alone and the box covers its middle; in the
dataset as described the image has a border of background and the box is the
sign, which is smaller at 64 pixels. Why the scores are lower was not tested.

Figures 2 to 6, the confusion matrices of the five experiments, are replaced
by those of the corrected experiments:
[Experiment 1](results/figures/confusion-as-described-exp1.pdf),
[2](results/figures/confusion-as-described-exp2.pdf),
[3](results/figures/confusion-as-described-exp3.pdf),
[4](results/figures/confusion-as-described-exp4.pdf),
[5](results/figures/confusion-as-described-exp5.pdf).
They count detections with a confidence of at least 0.25. In Experiment 2 no
such detection matched a sign, so every sign counts as not detected.

The training times given in Sections 3.1 to 3.5 are those of the 2023 runs.
The corrected runs shared GPUs, so their times are not comparable and none are
given.

## 4. What changes in the article

**What does not hold.**

- *Federated learning did not outperform centralised training* (abstract,
  Section 3.4). With four clients and ten local epochs it reached the accuracy
  of the centralised run: 0.389 against 0.363 on the dataset as described, and
  0.547 against 0.577 on the 2023 dataset. Two runs of the same settings
  differ by as much (see *Single runs* below). The abstract and Section 3.4
  should say that federated learning reached the accuracy of centralised
  training at its best setting, not that it outperformed it.
- *Forty global iterations instead of 17 gave more than "modest
  improvements"* (Section 3.5): the mAP@0.5 rose from 0.179 to 0.338.

**What holds.**

- *The accuracy of federated learning depends strongly on the experimental
  parameters*, as the article says. With sixteen clients and three local
  epochs the global model barely learns (0.068). With four clients it reaches
  0.179 after 51 epochs per client (Experiment 3), 0.338 after 120
  (Experiment 5) and 0.389 after 170 (Experiment 4).
- *More local training in a round can make up for fewer rounds* (Section
  3.4). After 120 epochs per client, ten local epochs a round, in 12 rounds,
  and three local epochs a round, in 40 rounds, give validation scores of
  0.352 and 0.335.

**What the corrected article adds.**

- *Test set.* The test images are other images of the signs in the training
  set (Section 1.3), so the scores do not measure the detection of new signs.
- *Accuracy.* The accuracy reached is low in every experiment: at a confidence
  of 0.25 the centralised model misses 51% of the test signs, and the best
  federated model 38%.
- *Single runs.* Each experiment was run once. Experiments 3 and 5 are the
  same procedure, with the same clients, for their first 17 rounds; after
  round 17 their validation mAP@0.5 was 0.177 and 0.224. The centralised
  experiment, run twice on the dataset as described (once with interruptions),
  scored 0.381 and 0.363. Differences of up to 0.05 between experiments should
  not be read as effects of the settings.
- *Computation.* In Experiment 4 the four clients each trained for 170 epochs
  on a quarter of the data, 3.4 times the computation of the centralised run's
  50 epochs on all of it. The centralised run's validation score was still
  rising when its 50 epochs ended.
- *The model a client returns.* Each client returns the local epoch that
  scored highest on the validation set, which all clients share and a client
  in a real federation would not have. In the rounds whose client records were
  kept, this was the first local epoch in 30 of 56 client runs of Experiment
  4, 74 of 116 of Experiment 5, 21 of 68 of Experiment 3 and 52 of 272 of
  Experiment 2; in those runs the later local epochs did not enter the
  average.
- *Learning-rate warm-up.* Every round starts YOLO v7's training afresh,
  warm-up included, and the warm-up lasts three epochs or 1,000 iterations,
  whichever is longer. With three local epochs a client's whole round is
  warm-up; with ten, seven epochs follow it. A comparison of three and ten
  local epochs is therefore also a comparison of learning rates.
- *One client alone.* The 2023 results were those of one client training
  alone. On the 2023 dataset, the same experiments with the averaging
  corrected come within 0.04 of them. The two sets of runs were made in
  different environments, in which the centralised experiment, which involves
  no averaging, scored 0.646 and 0.577, and no single-client run was made in
  2026. What averaging the clients' models adds over one client training
  alone was therefore not tested.

## 5. Data and code availability

The article has no data availability statement. The corrected article adds
this one.

The code, the original 2023 notebooks with their outputs, the results of
every experiment and the scripts that produce the figures are at
<https://github.com/Bennett-hash/fl-yolov7-road-signs>. GTSRB's home page is
<https://github.com/houbensebastian/GermanTrafficSignBenchmarks>. The
repository rebuilds, from the benchmark's official training archive, both the
dataset described in Section 2.1 and, exactly, the dataset the 2023 runs used.

## 6. Use of AI tools and authorship

I used an AI coding tool (Claude Code, Anthropic) to find and check these
errors in the 2023 code and data, to write the corrected implementation, to
run the experiments again, and to draft this note and the changes to the
article's text. I have checked the results and the text, and am responsible
for them. I am the article's only author.

I apologise to the article's readers for these errors.
