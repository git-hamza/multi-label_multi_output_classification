# multi-label_multi_output_classification

---

Task on Multi-label_multi-output classification is explained below.

## Approch Explained:
After reading the problem statement, we followed 3 steps to solve the problem i.e. `identifying problem`, 
`exploring dataset`, `model selection`, `code`.

### Identifying the problem:
After going through the problem statement and dataset, it was observed that multi output model was needed to solve this
problem. We had two major classes i.e. color and state (multi-output). However, different colors and states 
could appear at the same time which made it multi-label classification.

### Dataset exploration:
A total of `300` images/json were provided which is very low number for a deep neural network. After exploring 
the dataset, we observed that around `46` image did not have any color assigned to them. It would have made color
class empty, so we assign `no_color` label to those. Moreover, we converted all labels to lower case so Green and GREEN
could be counted as one green label.

There were total 16 unique labels in color class (including no_color) and 3 unqiue label in state class. 
Out of these *19*, *9* of had below 10 instance in the label. It was clearly seen that there is imbalance but due to the
time restriction we went for the modeling and kept the data as it is.
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
We did not go for custom modeling due to the small dataset. We wanted to use a pretrained model, so we can use their 
pretrained weights in order make up for the dataset. Bigger architecture like `ResNet` and `Inception` are not used as
the problem seems rather simple. We used pretrained mobilenetv2 and attach 2 classification layer to it
(color and state). For details, please look into the `model.py` file. Selection of loss and optimizer function
is also explained in the code files.

### Code:
Code structure is modular in nature.
For running the code, set configuration parameters and run the code as `train` or `test` by executing following command.
```bash
python3 main.py --mode train
```

- `main.py` main script which calls other modules
- `configuration.ini` configuration parameters can be setup here.
- `model.py` contains modeling details.
- `train.py` contains training loop.
- `test.py` used for model testing.
- `utils` have utility funcitons
  - `config.py` reading _configuration.ini_ and distributing it to other python modules
  - `helper_functions.py` have all the helper function used in the project.
- `dataset` have dataset related functions.
  - `data_attributes.py` used for data insights
  - `data_loading.py` loading the data into required format for model
  - `data_split.py` spliting the data between train, val and test.

Before running the code, please install packages fomr the `requirements.txt` file.

## Analysis
While triaining, we get the accuracy of combine color and state class as well with loss. Accuracy is a weak metric here for
evaluation due to the imbalance dataset and the multi labeled output. Although precision, recall and f1 score is also
extracted from the output.

![Training Graphs](graphs/loss_acc_graph.PNG?raw=true "Loss Accuracy Graph (training and validation)")

As the dataset was pretty low, so results on test testset was not very convincing. Howoever, metric for both trianing
and testing data `(thresh=0.5)` are shown below.

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

Testing:
```
Color Classification Report
              precision    recall  f1-score   support

      orange       0.00      0.00      0.00         0
       taupe       0.00      0.00      0.00         0
      silver       0.00      0.00      0.00         0
       black       0.00      0.00      0.00         0
       ochre       0.00      0.00      0.00         0
        gold       1.00      0.03      0.06        30
       umber       0.00      0.00      0.00         0
         red       0.00      0.00      0.00         0
      yellow       0.00      0.00      0.00         0
       white       0.00      0.00      0.00         0
       ivory       0.00      0.00      0.00         0
    no_color       0.00      0.00      0.00         0
        gray       0.00      0.00      0.00         0
       brown       0.00      0.00      0.00         0
       green       0.00      0.00      0.00         0
         tan       0.00      0.00      0.00         0

___________________________________
State Classification Report
              precision    recall  f1-score   support

     damaged       0.00      0.00      0.00         0
         old       0.00      0.00      0.00         0
         new       0.00      0.00      0.00        30

```