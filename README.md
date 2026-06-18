# Machine Learning Classification of Neuropsychiatric Disorders

Final project for the *Machine Learning and Big Data for Health* course. This work explores how machine learning can support the diagnosis of cognitive and psychiatric conditions starting from real clinical data collected through standardized neuropsychological tests (MMSE, Clock Drawing Test, Rey tests, FAB, and others).

The dataset describes 511 patients across 41 variables (demographic information, raw test scores, and qualitative outcomes) with a strongly imbalanced target where most subjects are healthy. The goal is to distinguish between four clinical groups: no pathology, mild cognitive impairment, depressive disorder, and Alzheimer's disease.

## Approach

The pipeline was built with care around the two main challenges of the dataset: heavy missingness and class imbalance.

- **Preprocessing**: removal of unreliable columns, outlier handling, and three custom scikit-learn transformers to impute missing values intelligently (using each test's qualitative outcome, plus a Random-Forest-based iterative imputer).
- **Balancing**: SMOTETomek (combined over- and under-sampling) to mitigate the dominance of the healthy class.
- **Modeling**: eight classifiers compared (Weighted KNN, Decision Tree, Balanced Random Forest, SVM, Extra Trees, LightGBM, CatBoost and XGBoost, each tuned via Grid Search with 5-fold stratified cross-validation).
- **Evaluation**: balanced accuracy, macro F1, and ROC AUC, chosen specifically because the dataset is imbalanced.

The problem was reframed at four levels of difficulty (four classes, three classes, binary, and pathological-only), and each was tested with three different feature subsets to study how variable selection affects performance.

## Key results

Difficulty scales clearly with the number of classes. Binary classification (disorder vs. no disorder) reached up to ~89% balanced accuracy with XGBoost, while the full four-class problem stayed around ~52%. Across all settings, Alzheimer's disease was the most reliably identified condition (a meaningful result given the clinical importance of its early detection).

## Tech stack

Python · scikit-learn · imbalanced-learn · XGBoost · CatBoost · LightGBM · pandas · Google Colab

## Documentation

Full methodology, figures, and results are available in `DocumentazioneProgetto.pdf`.
