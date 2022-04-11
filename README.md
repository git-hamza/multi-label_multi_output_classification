# Multi label multi output classification

---

Multi-label_multi-output classification approach is explained below.

## Approach:
For solving the task, 3 main steps were involved. `Problem identification`, `EDA`, `Model Selection`.
Finally, the coding part `code`.

### Problem identification:
After going through the problem statement, it is decided that a classification model with multiple output 
(multiple labels) is need of the hour. We have two major classes i.e. color and state (multi-output). 
However, different colors and states could appear at the same time which made it multi-label classification.

### EDA:
A total of `300` images/json were provided (a small number for a deep neural network). After exploring 
the dataset, we observed that `46` json did not have any color assigned to them. It would have made color
class empty, so we assigned `no_color` label to those. However, we converted all labels to lower case e.g.
Green and GREEN could be counted as one green label.

There were total 16 unique labels in color class (including no_color) and 3 unique labels in state class. 
Out of the total *19* classes, *9* of them have less than 10 instance. It was clearly seen that there is imbalance 
in the dataset but due to the time restriction we went for the modeling and kept the data as it is.

```
green: 68
gray: 36
yellow: 83
silver: 1
ivory: 1
ochre: 1
taupe: 1
orange: 11
no_color: 46
tan: 50
red: 1
gold: 1
brown: 197
white: 3
umber: 3
black: 19
new: 1
old: 285
damaged: 98
```
### Model Selection:
Custom model was out of option due to the size of dataset bottleneck. Instead, we went for a pretrained model to use
its pretrained weights to our advantage in order make up for the small dataset. Bigger architecture like `ResNet` 
and `Inception` are not used as the problem seems rather simple. We used pretrained *mobilenetv2* and attached 2 
classification layers to it(color and state). For details, please look into the `model.py` file. 
Selection of loss and optimizer function are also explained in the code files.

### Source code:
Code structure is modular in nature.
Set configuration parameters in *configuration.ini* and run the code as `train` or `test` by executing following command.
```bash
python3 main.py --mode train
```

- `main.py` main script which calls other modules
- `configuration.ini` configuration parameters can be set here
- `model.py` modeling details
- `train.py` training loop
- `test.py` for model testing
- `utils` utility functions
  - `config.py` reading _configuration.ini_ and distributing it to other python modules
  - `helper_functions.py` all the helper function used in the project.
- `dataset` dataset related functions.
  - `data_attributes.py` used for data insights
  - `data_loading.py` loading the data into required format for model
  - `data_split.py` splitting the data between train, val and test.

**Note:** Before running the code, please install packages from the `requirements.txt` file.

## Analysis
While training, we get the loss and accuracy of combine color and state class. Accuracy metric is a weak evaluation
metric in our case due to multi labeled output and imbalance dataset. Although precision, recall and f1 scores are also
extracted from the output.

![Training Graphs](graphs/loss_acc_graph.PNG?raw=true "Loss Accuracy Graph (training and validation)")

The dataset is pretty small, so results on test set was not good. However, metric for both training data `(thresh=0.5)` 
is shown below.

Training:
```
Color Classification Report
              precision    recall  f1-score   support

        gray       1.00      0.13      0.24       240
       ivory       0.00      0.00      0.00       240
      silver       0.00      0.00      0.00       240
      orange       1.00      0.05      0.09       240
       taupe       0.00      0.00      0.00       240
         red       1.00      0.00      0.01       240
       black       1.00      0.07      0.12       240
       white       1.00      0.01      0.02       240
         tan       1.00      0.17      0.30       240
       umber       1.00      0.00      0.01       240
       brown       1.00      0.65      0.79       240
       green       1.00      0.22      0.36       240
      yellow       1.00      0.28      0.43       240
    no_color       1.00      0.16      0.27       240
       ochre       0.00      0.00      0.00       240
        gold       1.00      0.00      0.01       240


___________________________________
State Classification Report
              precision    recall  f1-score   support

         new       1.00      0.00      0.01       240
         old       1.00      0.95      0.97       240
     damaged       1.00      0.33      0.49       240

```