# Airline Passenger Satisfaction — CatBoost + XGBoost Blend

This project predicts whether an airline passenger is satisfied. It was built for Kaggle's Playground Series S6E10 competition and compares a CatBoost model, an XGBoost model, and a probability blend of the two.

The competition has ended. The final recorded submission is the CatBoost–XGBoost blend. The Kaggle screenshots supplied with this project show a score of **0.95951**, a second score of **0.95851**, and leaderboard position **2,292**.

## Final result

| Item | Result |
|---|---:|
| Final submission file | submission_catboost_xgboost_blend.csv |
| Kaggle submission score shown first | **0.95951** |
| Kaggle submission score shown second | **0.95851** |
| Score displayed beside the leaderboard entry | **0.95951** |
| Final displayed leaderboard position | **2,292** |
| OOF AUC of the selected blend | **0.959093** |
| Blend weights | **50% CatBoost + 50% XGBoost** |

## Competition task

The labeled training data has **699,635 rows** and **23 columns**. It includes the satisfaction target and the id identifier. The test data has **299,844 rows** and **22 columns**: the same input fields and identifier, but no target.

The task is binary classification. Models output a probability that a passenger is satisfied. The competition metric is **ROC AUC**, which evaluates how well the model ranks satisfied passengers above dissatisfied passengers across probability thresholds. It is not ordinary accuracy at a fixed 0.5 cutoff.

The model inputs are the 21 passenger attributes. The id column was excluded from model training because it is a row identifier, not an assumed passenger attribute. The target is satisfaction.

## Dataset files

Place these files next to the notebooks:

| File | Purpose |
|---|---|
| train (1).csv | Labeled training data, including satisfaction |
| test.csv | Unlabeled competition test data |

The training data has 389,296 False labels and 310,339 True labels.

## Modeling approach

### Shared feature preparation

Both notebooks load the same training and test CSV files and use the same target and ID columns. Categorical values are prepared for the chosen model; numeric columns are coerced to numeric values while missing values are preserved for the tree models.

One engineered feature, Travel_Context, combines:

- Type of Travel
- Customer Type
- Class

This represents the combination of a passenger's travel purpose, customer category, and cabin class. It is added to both train and test data in both model workflows.

### CatBoost

CatBoost handles categorical columns directly. The final cross-validation setup uses:

| Parameter | Value |
|---|---:|
| Depth | 7 |
| Maximum iterations | 4,000 |
| Learning rate | 0.03 |
| Loss | Logloss |
| Evaluation metric | AUC |
| Early stopping patience | 150 |
| Fold seeds | 43–47 |

The notebook uses five-fold stratified cross-validation. Each fold model trains on four folds and predicts the held-out fifth fold. Those held-out predictions are saved as out-of-fold (OOF) probabilities. Each fold model also predicts the competition test rows; the five sets of test probabilities are averaged to create the CatBoost cross-validation submission.

### XGBoost

XGBoost receives numeric input. A one-hot encoder converts the categorical fields into indicator columns. The encoder is fitted only on each fold's training portion, then used to transform that fold's validation rows and the test data.

| Parameter | Value |
|---|---:|
| Maximum estimators | 4,000 |
| Learning rate | 0.03 |
| Maximum depth | 7 |
| Minimum child weight | 5 |
| Row subsampling | 0.85 |
| Column subsampling | 0.85 |
| L2 regularization (reg_lambda) | 5 |
| Tree method | hist |
| Evaluation metric | AUC |
| Early stopping patience | 150 |

XGBoost uses the same five-fold split seed as CatBoost so that their OOF probabilities refer to the same held-out rows. The five XGBoost test prediction sets are averaged for its cross-validation submission.

### Probability blend

The two OOF prediction files are joined by id. The notebook checks that IDs and target labels line up before comparing the models. It then tests blends from 0% to 100% XGBoost in 10% steps:

**blend = (1 − XGBoost weight) × CatBoost probability + XGBoost weight × XGBoost probability**

The best tested OOF blend assigned **50% weight to each model**. The same weights were then applied to the averaged test probabilities to produce submission_catboost_xgboost_blend.csv.

## Validation benchmarks

All three standalone/blend comparisons below use the five-fold OOF predictions on the labeled training data. AUC is reported to six decimal places.

| Model / blend | Five-fold mean AUC | Combined OOF AUC |
|---|---:|---:|
| CatBoost | **0.958897** | **0.958889** |
| XGBoost | **0.958944** | **0.958937** |
| 50/50 CatBoost–XGBoost blend | — | **0.959093** |

The fold-level mean and sample standard deviation were:

| Model | Fold AUCs | Mean ± sample SD |
|---|---|---:|
| CatBoost | 0.959005, 0.957759, 0.959677, 0.958841, 0.959202 | **0.958897 ± 0.000709** |
| XGBoost | 0.959060, 0.957943, 0.959692, 0.958867, 0.959161 | **0.958944 ± 0.000638** |

### Tested blend weights

The weight below is the fraction assigned to XGBoost; the remaining weight is assigned to CatBoost.

| XGBoost weight | CatBoost weight | OOF ROC AUC |
|---:|---:|---:|
| 0% | 100% | 0.958889 |
| 10% | 90% | 0.958962 |
| 20% | 80% | 0.959018 |
| 30% | 70% | 0.959057 |
| 40% | 60% | 0.959082 |
| **50%** | **50%** | **0.959093** |
| 60% | 40% | 0.959090 |
| 70% | 30% | 0.959074 |
| 80% | 20% | 0.959043 |
| 90% | 10% | 0.958998 |
| 100% | 0% | 0.958937 |

The blend's OOF AUC exceeded the better standalone OOF AUC by about **0.000157**.

## Files in the project

| File | Description |
|---|---|
| train.ipynb | CatBoost training, validation, five-fold CV, OOF predictions, and CatBoost submissions |
| train_xgboost.ipynb | XGBoost preprocessing, five-fold CV, OOF predictions, blend comparison, and blend submission |
| catboost_cv_results.csv | CatBoost fold AUCs and best tree counts |
| xgboost_cv_results.csv | XGBoost fold AUCs and best tree counts |
| catboost_oof_predictions.csv | Training-row IDs, labels, and CatBoost OOF probabilities |
| xgboost_oof_predictions.csv | Training-row IDs, labels, and XGBoost OOF probabilities |
| catboost_xgboost_blend_cv_results.csv | Tested blend weights and corresponding OOF AUCs |
| submission_catboost_cv.csv | Test predictions averaged across CatBoost fold models |
| submission_xgboost_cv.csv | Test predictions averaged across XGBoost fold models |
| submission_catboost_xgboost_blend.csv | Final 50/50 blend submission |
| submission_catboost.csv | Separate CatBoost full-data submission |

The two large source data files are train (1).csv and test.csv. Keep them in the project folder when running the notebooks.

## Reproducing the workflow

Install the required Python packages in the notebook environment: pandas, NumPy, scikit-learn, CatBoost, XGBoost, and Jupyter.

1. Open train.ipynb and run its cells in order through the five-fold cross-validation section. This creates the CatBoost OOF predictions, CV results, and fold-averaged test submission.
2. Open train_xgboost.ipynb and run its cells in order. Its final blend section reads the CatBoost outputs, checks row alignment, evaluates blend weights, and writes the final blend CSV.
3. Verify the final CSV has the expected columns, id and satisfaction, contains one row per test record, has unique IDs, and has probabilities between 0 and 1.
4. Submit submission_catboost_xgboost_blend.csv to Kaggle.

Training the five fold models takes substantially longer than fitting one model because each notebook trains five separate models and generates test predictions for each fold.

## Notes and interpretation

- The Kaggle result and the local OOF AUC measure performance on different data. A small difference between them is expected.
- The public leaderboard uses only a portion of the competition test set. Its score may differ from the final private score, and leaderboard rank can change as submissions are evaluated.
- The blend weight was selected from a coarse grid using OOF AUC. The 50/50 result was the best among the tested 10% increments, not a proof that it is the mathematically optimal weight.
- OOF results are estimates from these five folds and model settings; they are not a guarantee of performance on every unseen sample.
- The Kaggle screenshots are retained as the source for the final score and displayed ranking.

## Final takeaway

The project compared two gradient-boosted tree models with five-fold stratified CV, generated row-aligned OOF predictions, and selected a 50/50 probability blend. The blend achieved **0.959093 OOF ROC AUC** and the submitted file is recorded in the Kaggle screenshots with scores **0.95951** and **0.95851**. The leaderboard screenshot shows position **2,292** with score **0.95951**.
