# Breast Cancer Wisconsin 代码实现 in Python and Numpy

## How to run
```bash
pip install -r requirements.txt
```

Then run (make sure the data file `breast-cancer-wisconsin.txt` in the same directory with `project.py`, otherwise add param `--file_path your_datafile_path`)
```bash
python project.py --learning_rate 1e-3 --epochs 2000
```

The results will be like:
```bash
...
INFO: Epoch: 1999, Now loss: 8.3566
Figure saved to: ./loss.png
INFO: Evaluating model...
INFO: Accuracy on training dataset: 0.9728183118741058
INFO: Theoretical best w*: [[0.46484771]
 [0.32189252]
 [0.24762275]
 [0.08154626]
 [0.10405303]
 [0.66603437]
 [0.30151563]
 [0.25653716]
 [0.0343976 ]], Theoretical acc: 0.5512964566179767
INFO: Fitted best w*: [[ 0.25178853]
 [ 0.20825815]
 [ 0.23479147]
 [ 0.08345822]
 [-0.06529358]
 [ 0.35606041]
 [ 0.12270083]
 [ 0.13599248]
 [ 0.19858643]], Fitted best b*: -5.2674251411526045
INFO: Cosine similarity between theoretical w* and fitted w*: [[0.91393943]].
```
