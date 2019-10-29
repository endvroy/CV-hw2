## NYU-CV-Fall-2019

### Assignment 2: Traffic sign competition

#### Reproduction Instructions
1. Modify `data_prep.py` so that `zip_path` points to the zip file of the data
2. Run `data_prep.py`
3. Run `eval_test.py`. The test result is written to `submission.csv`

The model is `checkpoints/stn6/epoch_20.pth`. It is a `dict`. The weights are in `dict['model']`.