### Applied Machine Learning Analysis Project

##### Project Description

This project explores factors associated with airline passenger satisfaction using the Airline Passenger Satisfaction dataset. Using Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and SHAP, the project demonstrates a reproducible machine learning workflow including data cleaning, exploratory data analysis, feature preprocessing, model development, hyperparameter tuning, explainability analysis, and model evaluation.

The primary objective of the analysis is to predict whether a passenger is satisfied or neutral or dissatisfied based on demographic information, travel characteristics, flight delays, and service quality ratings. Multiple machine learning models were evaluated, including Logistic Regression and Random Forest, with Random Forest ultimately selected as the final model due to its superior predictive performance.

##### What I Built

- Loaded, validated, and explored the Airline Passenger Satisfaction dataset.
- Performed exploratory data analysis on passenger demographics, travel characteristics, service ratings, and delay variables.
- Built reusable preprocessing and machine learning pipelines using Scikit-learn.
- Developed and compared **Logistic Regression** and **Random Forest** classification models.
- Conducted feature scaling, log transformation, and hyperparameter tuning experiments.
- Evaluated model performance using Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices, and classification reports.
- Applied SHAP explainability analysis to identify the primary drivers of passenger satisfaction.
- Summarized key findings, limitations, and opportunities for future enhancement.

##### Dataset

Airline Passenger Satisfaction

Source (Kaggle):
[Kaggle Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

Files Used:

train.csv (103904, 25)
test.csv (25976, 25)

##### Dataset Characteristics

**Numerical Variables**
Unnamed:0
id
Age
Flight Distance
Departure Delay in Minutes
Arrival Delay in Minutes

**Categorical Variables**
Gender
Customer Type
Type of Travel
Class

**Ordinal Variables**
Inflight WiFi Service
Departure/Arrival Time Convenient
Ease of Online Booking
Gate Location
Food and Drink
Online Boarding
Seat Comfort
Inflight Entertainment
On-board Service
Leg Room Service
Baggage Handling
Check-in Service
Inflight Service
Cleanliness

**Target Variable**
Satisfaction

##### Bias Awareness and Limitations

As with all observational datasets, the results should be interpreted with caution. While the model achieved strong predictive performance, it identifies relationships between variables and passenger satisfaction rather than establishing causal relationships.

Potential limitations include:

- The dataset represents a specific airline passenger population and may not fully generalize to other airlines or geographic regions.
- The dataset does not contain potentially important variables such as ticket price, weather disruptions, airline-specific policies, or customer expectations.
- Delay-related variables exhibited substantial positive skew and required additional preprocessing investigation.
- Passenger preferences may change over time, creating potential model drift if deployed in future environments.

##### Future Extensions

Several opportunities exist to extend this analysis, like alternative model testing (XGBoost, Neural Networks) or:

`Fairness Assessment`

A useful extension would be to evaluate model performance across demographic and customer subgroups using dedicated Responsible AI and fairness assessment frameworks such as:

- Aequitas
- Fairlearn

`MLOps Pipeline Development`

Future work could include a complete MLOps workflow incorporating:

- Automated data validation
- Model versioning
- Experiment tracking
- Continuous retraining
- Performance monitoring

This would support deployment and long-term model maintenance.

##### How to Run the Project

1. Clone the Repository

```Shell
git clone <repository-url>
cd udacity-applied-ml-project
```

2. Create and Activate an Environment

Using Conda:

```Shell
conda env create -f environment.yml
conda activate statistical_analysis
```

Using venv:

```Shell
python -m venv statistical_analysis
```

Windows:

```Shell
statistical_analysis\Scripts\activate.bat
```

3. Install Dependencies

Python version:

`Python 3.11+`

Using pip:

```Shell
pip install -r requirements.txt
```

Using Conda:

```Shell
conda env create -f environment.yml
```

4. Open the Notebook

Launch Jupyter:

```Shell
jupyter notebook
```

Open:

`modeling.ipynb`

Run all cells from top to bottom.

#### Reproducibility

Generate the requirements file:

```Shell
pip freeze > requirements.txt
```

Generate the Conda environment specification:

```Shell
conda env export > environment.yml
```

The requirements.txt and environment.yml files are included in this repository to support reproducibility and environment recreation.
