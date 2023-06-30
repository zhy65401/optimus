# Optimus

## Pre-requisites
- Requires >= Python 3.6
- Set up a virtual environment and pip-install requirements.

## Quick start
- use `python setup.py bdist_egg` to build an egg file and upload to your jupyter and submit as other egg files.

```python
from optimus import Optimus
from apaylater_th_application_score.model import Model

model = Model([RESOURCE_ROOT])
model._load_resource()

o = Optimus.from_pipeline(model.resource['model'])
```
For other details, please proceed to https://confluence.advai.net/x/1Sl-BQ

## CHANGELOG
### v0.2.0
- Add EDA report generator. Now you can generate EDA report directly by calling `generate_eda_report`
- Add model performance comparison report. Now you can generate comparison report for multiple resources by calling `generate_eda_report` with `compare=True`.
- Remake the config properties. Now they look more structural. For more details, please check the docs.
- Add style (background colors) for eda and performance.
- Change the default binning strategy as `EDATransformer` from `bumblebee.eda`.
- Fix other bugs.

### v0.1.1
- Update the display mode to diagram for optmizing sklearn pipeline visualisation
- Move the property `TRAIN_CONFIG` from config to Optimus. So now you may use `o.TRAIN_CONFIG` to get this property instead of `o.config.TRAIN_CONFIG`.
- Fix other bugs

### v0.1.0
- Phase 0 and partial phase 1 features are finished.
- Init new repo
