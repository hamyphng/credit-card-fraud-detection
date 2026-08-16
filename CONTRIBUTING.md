# Contributing

Thanks for improving this fraud-detection project.

## Local setup

1. Create and activate the Python 3.11 virtual environment.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Put the dataset at `data/creditcard.csv`; do not commit it.
4. Run `python -m pytest -q` before opening a pull request.

## Development principles

- Keep validation and test sets in their original class distribution.
- Fit preprocessing steps only on training data.
- Apply resampling only inside the training portion of cross-validation.
- Report Recall, Precision, F1, and AUPRC; do not use Accuracy as the primary metric.
- Do not tune model settings after inspecting final test metrics.

## Pull requests

- Keep each pull request focused.
- Explain the metric impact and any trade-off between false positives and false negatives.
- Add or update tests for source-code changes.
- Do not commit datasets, model binaries, credentials, or notebook checkpoints.
