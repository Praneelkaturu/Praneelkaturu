# Titanic Logistic Regression Case Study

## Files
- `Titanic_Logistic_Regression_Case_Study.ipynb`: notebook with explanations, analysis, plots, model evaluation, and coefficient interpretation.
- `titanic_logistic_regression.py`: standalone training/evaluation script.
- `titanic.csv`: dataset required by both files.
- `requirements.txt`: Python dependencies.

## Run locally
1. Put all files in one folder.
2. Install Python 3.10+.
3. Open a terminal in the folder and run `python -m pip install -r requirements.txt`.
4. Start Jupyter with `jupyter notebook`.
5. Open the notebook and choose **Kernel > Restart & Run All**.
6. Verify all cells run and outputs are visible before saving/submitting.

## Notes
- The notebook uses a stratified 80/20 train/test split with `random_state=42`.
- Preprocessing is fitted on training data only via a scikit-learn pipeline.
- Update your report's title-page placeholders and write interpretations based on your actual outputs.
