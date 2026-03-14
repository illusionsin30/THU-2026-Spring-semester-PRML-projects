# Face Recognition Implementation in Python with Sklearn

## Original Question
See the raw homework file `lec3_homework.pdf`.

## How to run
```bash
pip install -r requirements.txt
```

Then run (make sure the data file `face_data` in the same directory with `project.py`, otherwise add param `--data_dir your_datafile_path`)
```bash
python project.py
```

The results will be like:
```bash
INFO: Experiment settings:                  
 kernel: linear, C: 1.0, degree: 3                  
 tol: 0.001, max_iteration: -1, seed: 0.
INFO: Training model...
INFO: Finished training. Start Evaluate.
INFO: The accuracy of model on test data: 0.9700.
INFO: Plotting figures.
INFO: Figure has been saved at ./imgs/sv_linear_1.0.png.
```

## Params setting

|Param|Choices|
|:---:|:---:|
|`--kernel`| `linear`, `sigmoid`, `rbf`, `poly` |
|`--C`| any float in range (0.0, 1.0] |
|`--seed`| any int |
|`--degree`| any int, requires `--kernel` set as `poly`|