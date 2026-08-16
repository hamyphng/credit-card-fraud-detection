# Dataset

This repository does not include the raw dataset because `creditcard.csv` exceeds GitHub's file-size limit.

Download the European Credit Card Fraud Detection dataset from Kaggle and place the CSV at:

```text
data/creditcard.csv
```

The notebooks expect the columns `Time`, `V1` through `V28`, `Amount`, and `Class`.

Do not commit the downloaded CSV. It is excluded by `.gitignore`.
