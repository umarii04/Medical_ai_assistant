# Medical AI Assistant

A machine learning-powered assistant for disease prediction based on user symptoms, with a user-friendly Streamlit web interface.

## Project Structure

```
Medical_AI_Assistant/
├── data/
│   ├── dataset.csv
│   ├── final.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   ├── Symptom-severity.csv
│   ├── Cleaned_Symptom_description.csv
│   ├── Cleaned_Symptom_precautions.csv
│   ├── Cleaned_Symptom_severity.csv
├── docs/
├── models/
│   └── best_ml_combined_model.pkl
├── notebooks/
│   ├── Cleaned_Dataset,working.ipynb
│   ├── Cleaned_Dataset.ipynb
│   ├── Symptom_descriptive_cleaned.csv
│   ├── Symptom_precautions_cleaned.csv
│   ├── Symptom_severity_cleaned.csv
│   └── symptom_index.pkl
├── src/
├── Streamlit/
│   └── app.py
├── requirements.txt
└── README.md
```

## Features
- Data cleaning, preprocessing, and exploratory analysis in Jupyter notebooks
- Machine learning model training for disease prediction
- Model and symptom index serialization for deployment
- Streamlit web app for user-friendly disease prediction
- Drug, precaution, and severity information for predicted diseases

## How to Use

### 1. Data Preparation & Model Training
- Use the Jupyter notebook (`notebooks/Cleaned_Dataset,working.ipynb`) to:
  - Clean and preprocess the dataset
  - Train a machine learning model
  - Save the trained model as `models/best_ml_combined_model.pkl`
  - Save the symptom index as `notebooks/symptom_index.pkl`

### 2. Running the Streamlit App
- Install requirements:
  ```bash
  pip install -r requirements.txt
  ```
- Run the app:
  ```bash
  streamlit run Streamlit/app.py
  ```
- Open your browser at [http://localhost:8501](http://localhost:8501)
- Enter symptoms (comma separated) to get disease predictions and recommendations

### 3. Project Files
- **data/**: Raw and cleaned datasets, symptom descriptions, precautions, and severity
- **models/**: Trained machine learning models
- **notebooks/**: Jupyter notebooks for data analysis and model training
- **Streamlit/app.py**: Streamlit web application
- **requirements.txt**: Python dependencies

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, joblib, streamlit

## Authors
- [Muhammad Umar Farooqi](https://github.com/umarii04)

## License
This project is for educational and research purposes only.
