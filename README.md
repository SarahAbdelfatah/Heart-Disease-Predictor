# Heart Disease Prediction Using Machine Learning

A machine learning classification project exploring the prediction of heart disease using clinical attributes. The project compares tree-based classification algorithms, investigates overfitting, and evaluates hyperparameter tuning techniques to improve model generalization.

## Project Overview

The objective is to predict the presence or absence of heart disease using 13 clinical features, including age, cholesterol, blood pressure, chest pain type, and maximum heart rate.

The notebook follows a machine learning workflow covering exploratory data analysis, preprocessing, model training, optimization, and evaluation.

**Target variable:**
- `0`: No heart disease
- `1`: Heart disease

## Technologies Used

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

## Dataset

The dataset contains 1,025 records and 13 input features, plus a binary target variable.

During preprocessing:
- Checked data types, feature distributions, and missing values.
- Identified and removed 723 duplicate rows, retaining 302 unique records.
- Applied one-hot encoding to multicategory nominal features.
- Split the processed data into training (80%) and testing (20%) sets.

The original CSV is loaded as `heart.csv` in the notebook.

## Models Implemented

### 1. Decision Tree Classifier

Compared Gini impurity and entropy as splitting criteria.

Investigated overfitting by comparing training and testing accuracy, then applied pre-pruning through `max_depth` and `min_samples_leaf`.

### 2. Random Forest Classifier

Trained an ensemble of decision trees and experimented with maximum tree depth and the number of estimators.

The selected model used:
- Criterion: Entropy
- Maximum depth: 7
- Number of estimators: 109

### 3. AdaBoost Classifier

Applied boosting and used `RandomizedSearchCV` with 5-fold cross-validation to explore combinations of learning rate and estimator count.

## Results

| Model | Test Accuracy |
|---|---|
| Decision Tree (Gini) | 70.49% |
| Decision Tree (Entropy) | 78.69% |
| Pruned Decision Tree (Entropy) | 80.33% |
| Random Forest | **83.61%** |
| Tuned AdaBoost | 78.69% |

Random Forest achieved the highest reported test accuracy among the evaluated models.

## Key Observations

- Unrestricted Decision Trees achieved 100% training accuracy but showed substantially lower performance on unseen test data, indicating overfitting.
- Restricting tree complexity reduced the training-testing performance gap.
- Random Forest achieved the strongest reported test accuracy.
- AdaBoost hyperparameter tuning did not outperform Random Forest in this experiment.
- Removing duplicate rows before splitting helped prevent identical records from appearing in both training and testing sets.

## How to Run

1. Clone or download this repository.
2. Install the required dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Place the dataset file `heart.csv` in the same working directory as the notebook.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Heart-Disease-Predictor.ipynb` and execute the cells sequentially.

## Limitations

- The dataset contains only 302 unique records after duplicate removal.
- Some hyperparameter experiments select configurations using test-set performance, which can introduce selection bias.
- The reported accuracy is based on a single train-test split rather than a fully independent final evaluation.
- Model performance on this dataset does not establish clinical reliability or generalizability to real patient populations.

**Disclaimer:** This project is intended for educational and experimental purposes only. It is not a medical diagnostic tool.

## Future Improvements

- Use stratified train-test splitting and cross-validation.
- Perform hyperparameter selection exclusively on training data.
- Evaluate precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Compare additional baseline classifiers.
- Evaluate the final selected model on an untouched test set.
