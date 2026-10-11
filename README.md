# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave the service (churn). The project includes model experimentation in a Jupyter notebook and a Flask web application for entering customer details and viewing a prediction.

## Project Features

- Explores telecom customer churn data using Python and Pandas.
- Compares three classification algorithms: Logistic Regression, Random Forest, and Decision Tree.
- Evaluates models using accuracy, precision, recall, and F1-score.
- Saves the trained model and supporting preprocessing artifacts with Joblib.
- Provides a Flask web interface for submitting customer details and viewing a churn prediction.

## Technologies Used

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Web application:** Flask, HTML, CSS
- **Development tools:** Jupyter Notebook, Visual Studio Code, Git, GitHub

## Model Evaluation

These are the results previously recorded during model evaluation. They depend on the dataset and test split used.

| Algorithm | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 78.82% | 62.34% | 51.34% | 56.30% |
| Decision Tree | 71.64% | 46.61% | 45.99% | 46.30% |
| Logistic Regression | 80.38% | 64.76% | 57.49% | 60.91% |

Logistic Regression had the highest accuracy and F1-score among these three recorded results. Confirm that the model loaded by `app.py` is the same model that produced these results before describing them as the deployed model's performance.

## Project Structure

```text
Customer-Churn-Prediction/
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── notebook/
│   └── customer_churn.ipynb
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The exact files may vary. Keep model files listed above only if your application uses them.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Nandhakishore1710/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the Flask app

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

> The saved model and preprocessing files must exist at the paths expected by `app.py`. If a file is missing, export the required artifact or correct the paths before running the app.

## How It Works

1. Customer churn data is prepared for machine learning.
2. Classification algorithms are trained and evaluated on training/test data.
3. The selected model and any required preprocessing objects are saved.
4. The Flask app accepts customer details through an HTML form.
5. The app applies the preprocessing expected by the trained model and displays a prediction.

## Important Notes

- The form must use the same feature names, order, encoding, and preprocessing used during training.
- Do not fit the scaler again on a single user input. Use the scaler fitted during training if scaling is required.
- Predictions are estimates based on training data, not guarantees about a customer's future behavior.
- Before making the repository public, verify that the CSV is an appropriate public/sample dataset and contains no private customer information.
- Never commit virtual environments, credentials, API keys, or confidential customer data.

## Future Improvements

- Add input validation and helpful error messages.
- Add automated tests for preprocessing and prediction routes.
- Improve the interface and include model evaluation visualizations.
- Deploy the application to a hosting platform.
- Monitor model performance and retrain when appropriate.

## Author

**Nandhakishore1710**

GitHub: [Nandhakishore1710](https://github.com/Nandhakishore1710)