JSTARS Automated Discovery
================

[![Formatted with black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Linted with pylint](https://img.shields.io/badge/linting-pylint-green)](https://github.com/PyCQA/pylint)


<img width="256" alt="ab" src="https://github.com/lesnikow/jstars-automated-discovery/assets/8730814/90c589e3-168f-4863-8a7e-06b6fef172e7">
<img width="256" alt="cd" src="https://github.com/lesnikow/jstars-automated-discovery/assets/8730814/e15ecfd0-8d07-435f-8888-31e805c0fdc8">
<img width="256" alt="ef" src="https://github.com/lesnikow/jstars-automated-discovery/assets/8730814/8f861370-f4e1-4d1d-8dcb-5465bb25a476">

A machine learning-based method for finding scientifically-relevant lunar anomalies such as landed assets, volcanic pit skylights, irregular mare patches, recent impacts, and rockfalls. 

----
For complete details, see ["Automated Discovery of Anomalous Features in Ultralarge Planetary Remote-Sensing Datasets Using Variational Autoencoder"](https://doi.org/10.1109/JSTARS.2024.3369101) in the [IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing](https://www.grss-ieee.org/publications/journal-of-selected-topics-in-applied-earth-observations-and-remote-sensing/0).

The DOI for this paper is [https://doi.org/10.1109/JSTARS.2024.3369101](https://doi.org/10.1109/JSTARS.2024.3369101).

----


## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pre-trained model download

There is a pre-trained model .pt file not being tracked in git due to it being a large binary file.
This model file is needed to run e.g. `evaluate.py` without having to train your own model. 
The plan is to have `evaluate` download do it automatically as a default, if needed.
In the meantime, [download the model here](https://drive.google.com/file/d/1m3xrnoFm0gyK_UUklZmpUGc5flqpAyuw/view?usp=sharing) and manually add this `.pt` file to the `models/` directory in your local copy of this repo to have this pre-trained model available to `evaluate`.

## Sample commands
To download raw images with default settings on the Apollo 17 'ap17' feature, run:
```bash
python lam/evaluate.py --feature_str ap17 --download_raws
```
To inference a model with default settings on the Apollo 17 'ap17' feature, and generate
anomlay scores, run:
```bash
python lam/evaluate.py --feature_str ap17 --inference
```
To generate a precision-recall curve and a KDE plot for the 'crater' feature:
```bash
python lam/evaluate.py --feature_str crater --make_pr_curve --make_kde_plot
```
To perform a KS test on anomaly scores for the 'rockfall' feature:
```bash
python lam/evaluate.py --feature_str rockfall --make_ks_test
```

### Known issues

There are currently hard-coded paths in a couple areas of the codebase, e.g. in `evaluate.py`. 
The plan is to update these hard-coded paths either to sensible generic defaults or user-inputted paths.
If you're feeling up for it, you can update these paths for your own local setup.
In the meantime, watch out for updates here to make setup and runs easier for new users. 




