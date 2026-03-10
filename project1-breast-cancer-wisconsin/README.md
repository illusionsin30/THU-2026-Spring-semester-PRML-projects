# Breast Cancer Wisconsin 代码实现 in Python and Numpy

## How to run
```bash
pip install -r requirements.txt
```

Then run (make sure the data file `breast-cancer-wisconsin.txt` in the same directory with `project.py`, otherwise add param `--file_path your_datafile_path`)
```bash
python project.py --epochs 1000
```

The results will be like:
```bash
...
INFO: Epoch: 999, Now loss: 7.2658
Figure saved to: ./loss.png
INFO: Evaluating model...
INFO: Accuracy on test dataset: 1.0
INFO: Theoretical best w*: [[0.46484771]
 [0.32189252]
 [0.24762275]
 [0.08154626]
 [0.10405303]
 [0.66603437]
 [0.30151563]
 [0.25653716]
 [0.0343976 ]], Theoretical acc: 0.5512964566179767
INFO: Fitted best w*: [[0.47139569]
 [0.15440241]
 [0.29066424]
 [0.19109434]
 [0.00279701]
 [0.46015713]
 [0.23155275]
 [0.12121683]
 [0.44690223]], Fitted best b*: -7.7811479899554135
INFO: Cosine similarity between theoretical w* and fitted w*: [[0.84700065]].
```
