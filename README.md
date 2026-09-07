# 🏠 House Price Predictor

### An End-to-End Machine Learning Project for House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple.svg)](https://dvc.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)

> My first end-to-end Data Science and Machine Learning project, built to understand how a machine learning model moves from raw data to a reproducible training pipeline.

---

## 📌 Overview

**House Price Predictor** is an end-to-end Machine Learning project that demonstrates the complete workflow involved in developing a regression-based predictive system.

The project focuses on building a structured ML pipeline rather than only training a model inside a Jupyter Notebook.

The workflow includes:

* Data ingestion
* Data preprocessing
* Feature transformation
* Model training
* Model evaluation
* Model monitoring
* Model serialization
* Experiment tracking
* Data versioning
* Reproducible ML workflows

The project is organized into separate components and pipelines to make the code easier to maintain, debug, and extend.

---

## 🎯 Project Objective

The primary objective is to develop a machine learning system capable of learning patterns from historical housing data and using those patterns to estimate house prices.

The project also aims to demonstrate practical Data Science and MLOps concepts such as:

**Raw Data → Data Ingestion → Data Transformation → Model Training → Model Evaluation → Model Artifact**

---

## 🧠 Machine Learning Workflow

```text
                    ┌─────────────────┐
                    │    Raw Data     │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Data Ingestion    │
                  └──────────┬──────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │ Data Transformation  │
                 │   & Preprocessing    │
                 └──────────┬───────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │   Model Training   │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Model Evaluation   │
                  └─────────┬──────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Trained Model +     │
                 │    Preprocessor      │
                 └──────────┬───────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Prediction System  │
                  └────────────────────┘
```

---

## 🛠️ Tech Stack

| Technology                    | Purpose                                         |
| ----------------------------- | ----------------------------------------------- |
| **Python**                    | Core programming language                       |
| **Pandas**                    | Data manipulation and analysis                  |
| **NumPy**                     | Numerical computation                           |
| **Scikit-learn**              | Machine Learning and preprocessing              |
| **CatBoost**                  | Gradient boosting                               |
| **XGBoost**                   | Gradient boosting                               |
| **SciPy**                     | Scientific computing                            |
| **Statsmodels**               | Statistical analysis                            |
| **Matplotlib**                | Data visualization                              |
| **Seaborn**                   | Data visualization                              |
| **MLflow**                    | Experiment tracking and ML lifecycle management |
| **DVC**                       | Dataset/version management                      |
| **DAGsHub**                   | ML experiment/data collaboration                |
| **Dill**                      | Python object serialization                     |
| **Flask**                     | Web application/API foundation                  |
| **MySQL Connector / PyMySQL** | Database connectivity                           |
| **Jupyter / IPykernel**       | Interactive experimentation                     |

---

## 📂 Project Structure

```text
House_Price_predictor/
│
├── .dvc/
│
├── .github/
│   └── workflows/
│
├── Notebook/
│   └── # Jupyter notebooks and experimentation
│
├── Source/
│   └── ML_codes/
│       │
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_monitoring.py
│       │
│       ├── pipelines/
│       │   └── training_pipeline.py
│       │
│       ├── exception.py
│       ├── logger.py
│       └── utils.py
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   ├── test.csv
│   └── raw.csv.dvc
│
├── catboost_info/
│
├── mlruns/
│   └── # MLflow experiment tracking data
│
├── app.py
├── Dockerfile
├── requirement.txt
├── setup.py
├── template.py
├── .gitignore
├── .dvcignore
└── README.md
```

---

## 🔄 Pipeline Components

### 1. Data Ingestion

The data ingestion component is responsible for obtaining the raw dataset and preparing it for the next stage of the machine learning pipeline.

The process produces datasets that can subsequently be used for training and testing.

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ├── Training Data
     │
     └── Testing Data
```

---

### 2. Data Transformation

The transformation stage prepares the data for machine learning.

Typical responsibilities include:

* Data cleaning
* Handling numerical features
* Handling categorical features
* Feature preprocessing
* Feature transformation
* Preparing data for model training

The fitted preprocessing object is stored as:

```text
artifacts/preprocessor.pkl
```

---

### 3. Model Training

The model-training component trains regression models using the transformed dataset.

The repository includes support for modern tree-based boosting approaches such as:

* CatBoost
* XGBoost

along with the broader Scikit-learn ecosystem.

The trained model is serialized as:

```text
artifacts/model.pkl
```

---

### 4. Model Evaluation

The trained model is evaluated to determine how well it performs on unseen test data.

For a regression problem, relevant evaluation metrics include:

* MAE — Mean Absolute Error
* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

Lower MAE/RMSE and higher R² generally indicate better predictive performance.

---

### 5. Model Monitoring

The project also contains a dedicated model monitoring component.

This provides a foundation for tracking model behavior after training and can be extended toward production monitoring and model-drift detection.

---

## 📦 Model Artifacts

The `artifacts/` directory contains important outputs generated during the ML pipeline.

```text
artifacts/
│
├── model.pkl
├── preprocessor.pkl
├── train.csv
├── test.csv
└── raw.csv.dvc
```

### `model.pkl`

Serialized trained machine learning model.

### `preprocessor.pkl`

Serialized preprocessing pipeline required to transform input data before prediction.

### `train.csv`

Training dataset generated during data preparation.

### `test.csv`

Testing dataset used for evaluating model performance.

### `raw.csv.dvc`

DVC metadata used to track/version the raw dataset.

---

## 📊 Experiment Tracking with MLflow

MLflow is included in the project for experiment tracking.

It can be used to record:

* Model parameters
* Evaluation metrics
* Experiment runs
* Model information
* Training results

The repository contains an `mlruns/` directory for MLflow tracking data.

This allows different experiments to be compared instead of relying only on manually recorded results.

---

## 🗃️ Data Version Control with DVC

The project uses **DVC (Data Version Control)** to manage data separately from normal Git version control.

This is particularly useful for machine learning projects because datasets can be large and may change throughout development.

The project includes:

```text
.dvc/
.dvcignore
artifacts/raw.csv.dvc
```

DVC helps make the ML workflow more reproducible by keeping track of which version of the dataset was used.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MitadruMridha05/House_Price_predictor.git
```

Move into the project directory:

```bash
cd House_Price_predictor
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirement.txt
```

---

## ▶️ Running the Project

The current `app.py` acts as the main executable entry point for running the machine learning pipeline.

Run:

```bash
python app.py
```

The pipeline performs the following operations:

```text
Data Ingestion
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Artifact Generation
```

---

## 🧪 Running the Training Pipeline

The main training pipeline can also be found under:

```text
Source/ML_codes/pipelines/training_pipeline.py
```

The project follows a modular architecture so individual stages can be developed and tested independently.

---

## 🐳 Docker

A `Dockerfile` is included in the repository to provide a foundation for containerizing the project.

Build the Docker image:

```bash
docker build -t house-price-predictor .
```

Run the container:

```bash
docker run house-price-predictor
```

> Docker configuration may require further customization depending on the desired deployment architecture.

---

## 🧪 Development Workflow

A typical development workflow for this project is:

```text
1. Collect / update dataset
          ↓
2. Version dataset using DVC
          ↓
3. Perform exploratory data analysis
          ↓
4. Build preprocessing pipeline
          ↓
5. Train multiple ML models
          ↓
6. Evaluate models
          ↓
7. Track experiments with MLflow
          ↓
8. Select the best model
          ↓
9. Save model + preprocessor
          ↓
10. Deploy / serve predictions
```

---

## 📈 Future Improvements

This project is a learning-oriented implementation and can be extended into a more production-ready ML system.

Potential improvements include:

* [ ] Build a complete prediction API using Flask/FastAPI
* [ ] Add a frontend for user interaction
* [ ] Add automated CI/CD with GitHub Actions
* [ ] Improve Docker deployment
* [ ] Add automated model testing
* [ ] Add model performance monitoring
* [ ] Add data-drift detection
* [ ] Add automated retraining
* [ ] Deploy the application to a cloud platform
* [ ] Add REST API documentation
* [ ] Add unit and integration tests
* [ ] Improve experiment tracking with MLflow
* [ ] Integrate DVC with a remote data store
* [ ] Add production-grade logging
* [ ] Add model versioning and model registry

---

## 🎓 Learning Outcomes

This project helped me understand the practical implementation of:

* Machine Learning workflows
* Regression problems
* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Python project structuring
* Modular ML architecture
* Exception handling
* Logging
* Model serialization
* Experiment tracking
* Data version control
* Basic MLOps concepts

---

## 🚀 Why This Project?

This project was created as my **first Data Science / Machine Learning project**.

Instead of stopping at:

```python
model.fit(X_train, y_train)
```

the goal was to understand how a machine learning project can be organized into reusable components and gradually transformed into an end-to-end ML system.

It serves as a foundation for learning more advanced topics such as:

```text
Machine Learning
       ↓
Model Deployment
       ↓
MLOps
       ↓
CI/CD
       ↓
Cloud
       ↓
Production ML Systems
```

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

To contribute:

```bash
git clone https://github.com/MitadruMridha05/House_Price_predictor.git
cd House_Price_predictor
```

Create a new branch:

```bash
git checkout -b feature/your-feature
```

Make your changes, commit them, and push the branch:

```bash
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
```

Then open a Pull Request.

---

## 📜 License

This project is intended primarily for educational and learning purposes.

If you add a formal license to the repository, update this section accordingly.

---

## 👨‍💻 Author

**Mitadru Mridha**

GitHub: [@MitadruMridha05](https://github.com/MitadruMridha05)

---

## ⭐ Acknowledgement

This project represents my first step into practical Data Science and Machine Learning.

It was built to move beyond theoretical ML concepts and gain hands-on experience with:

**Data → ML → Engineering → MLOps**

If you find the project useful or interesting, consider giving this repository a ⭐.
