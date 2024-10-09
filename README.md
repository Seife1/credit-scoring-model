# Credit Scoring Model for Bati Bank

This repository contains a Credit Scoring Model built for Bati Bank as part of the Week 6 Challenge at 10 Academy. The goal is to develop a machine learning model that assesses the credit risk of customers using their transaction data from an eCommerce platform. The model predicts whether a customer is likely to default on a loan and also provides a credit score based on risk probability estimates.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [API Deployment](#api-deployment)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Bati Bank is partnering with an eCommerce platform to provide a buy-now-pay-later service. The aim is to assess the creditworthiness of customers and predict the likelihood of default on future loans. This project involves:
1. Defining a proxy variable for default risk.
2. Performing feature engineering to extract key features from transaction data.
3. Developing models to predict the credit score and the optimal loan amount.
4. Serving the model via a REST API for real-time predictions.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Seife1/credit-scoring-model.git
   cd credit-scoring-model
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```bash
root/
│
├── api/                        # Code for serving the model via REST API
│ ├── models/                     # Trained models
├── data/                       # Raw and processed data
├── logs/                       # Logs from model training and API deployment
├── notebooks/                  # Jupyter notebooks for EDA and modeling
├── scripts/                    # Python scripts for feature engineering and model training
├── tests/                      # Unit tests and integration tests
└── README.md                   # Project overview (this file)
```

## Data
The dataset is provided by the eCommerce platform and contains customer transaction data. More details on the dataset can be found [here](https://www.kaggle.com/datasets/atwine/xente-challenge)

## Usage
**Exploratory Data Analysis (EDA)**: The notebooks/1_EDA.ipynb contains initial data exploration and visualization.

**Feature Engineering**: The notebooks/2_Feature_Engineering.ipynb contains derived features that provide insights about each customers.

**Model Training**: A proxy variable is created based on RFMS formalism, Binning based on Weight of Evidence (WoE), script to train the models and fine tune parameters by hyperparameter tuning.

**API Deployment**: A RESTful API is developed to serve the trained credit scoring model for real-time predictions:

- **Framework**: Flask.
- **Endpoints**: API accepts transaction data as input and returns credit risk predictions (Good/Bad) and credit scores.

## Modeling Approach
The project explores several machine learning models, including:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting Machines
The models are evaluated using metrics such as Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

## API Deployment
The trained model is deployed as a REST API using `Flask`. It can accept customer data and return credit risk predictions. More details on how to make API calls are provided in the api/ folder.

## Contributing
If you would like to contribute to this project, please submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.