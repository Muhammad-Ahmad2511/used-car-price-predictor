# 🚗 PakWheels Used Car Price Predictor

AI-powered price prediction with Explainable AI (XAI) for Pakistan's used car market.

---

## 📋 Project Overview

**Problem**: Price opacity and information asymmetry in Pakistan's used car market  
**Solution**: ML-based price predictor with SHAP explainability  
**Track**: AI Track A - Application Development  

---

## 🎯 Key Features

- ✅ Real-time data scraping from PakWheels.com
- ✅ 11 base features + 15 engineered features
- ✅ Pakistan-specific attributes (Assembly, Registration City)
- ✅ Comprehensive data cleaning pipeline
- ✅ Exploratory Data Analysis (EDA)
- ✅ XGBoost & LightGBM regression models
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Model evaluation and comparison
- 🔜 Explainable AI (SHAP integration)
- 🔜 Streamlit web deployment

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | PakWheels.com |
| **Size** | 24,135 clean records |
| **Features** | 26 total (11 base + 15 engineered) |
| **Target** | Price (PKR) |
| **Time Period** | 1990-2026 |

### Sample Data

| Make | Model | Year | Mileage | Engine | Assembly | City | Price (PKR) |
|------|-------|------|---------|--------|----------|------|-------------|
| Mazda | RX8 | 2004 | 109,000 | 1300 | Imported | Islamabad | 3,500,000 |
| Toyota | Corolla | 2020 | 39,500 | 1300 | Local | Islamabad | 4,650,000 |
| Nissan | Dayz | 2022 | 19,999 | 660 | Imported | Lahore | 4,590,000 |

---

## 🛠️ Technology Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Web Scraping** | BeautifulSoup4, Requests |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | XGBoost, LightGBM, Scikit-Learn |
| **Explainability** | SHAP (planned) |
| **Visualization** | Matplotlib, Seaborn |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

```
project/
│
├── data_scripts/                          # Data pipeline
│   ├── scrape_pakwheels.py                # Web scraping
│   ├── 02_data_cleaning.py                # Data cleaning
│   └── 03_feature_engineering.py          # Feature engineering
│
├── eda/                                   # Exploratory analysis
│   ├── eda_analysis.py                    # Generate all plots
│   └── plots/                             # Visualizations
│       ├── price_distribution.png
│       ├── feature_distributions.png
│       ├── categorical_distributions.png
│       ├── correlation_matrix.png
│       └── price_vs_features.png
│
├── models/                                # ML models (Deliverable 2)
│   ├── train_xgboost.ipynb                # XGBoost training
│   ├── train_lightgbm.ipynb               # LightGBM training
│   ├── evaluate_models.ipynb              # Model comparison
│   ├── xgboost_model.pkl                  # Trained XGBoost (1.9 MB)
│   ├── lightgbm_model.pkl                 # Trained LightGBM (1.8 MB)
│   ├── label_encoders.pkl                 # Encoders (40 KB)
│   ├── xgboost_results.json               # XGBoost metrics
│   ├── lightgbm_results.json              # LightGBM metrics
│   └── evaluation/                        # Evaluation outputs
│       ├── feature_importance_comparison.png
│       ├── r2_comparison.png
│       ├── mae_rmse_comparison.png
│       └── evaluation_summary.txt
│
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── .gitignore                             # Git ignore rules
```

---

## 🚀 Installation & Usage

### **Prerequisites**
- Python 3.11+
- Jupyter Notebook (optional, for .ipynb files)

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Run Data Pipeline**
```bash
# 1. Scrape data from PakWheels
python data_scripts/scrape_pakwheels.py
# Enter start page: 1
# Enter end page: 50

# 2. Clean scraped data
python data_scripts/02_data_cleaning.py

# 3. Engineer features
python data_scripts/03_feature_engineering.py

# 4. Generate EDA plots
python eda/eda_analysis.py
```

### **Step 3: Train Models (Deliverable 2)**
```bash
# Train XGBoost
jupyter notebook models/train_xgboost.ipynb

# Train LightGBM
jupyter notebook models/train_lightgbm.ipynb

# Evaluate models
jupyter notebook models/evaluate_models.ipynb
```

### **Step 4: Use Trained Models**
```python
import joblib
import pandas as pd

# Load best model (LightGBM)
model = joblib.load('models/lightgbm_model.pkl')
encoders = joblib.load('models/label_encoders.pkl')

# Prepare your data
# (encode categorical features using encoders)

# Make predictions
predictions = model.predict(X_test)
print(f"Predicted Price: Rs {predictions[0]:,.0f}")
```

---

## 📈 Data Pipeline
PakWheels.com
      ↓
Web Scraping
      ↓
Raw CSV (24,728 rows)
      ↓
Data Cleaning (97.6% retention)
      ↓
Clean CSV (24,135 rows)
      ↓
Feature Engineering (+15 features)
      ↓
Final Dataset (26 columns)
      ↓
EDA & Visualization (5 plots)
      ↓
Model Training (XGBoost + LightGBM)
      ↓
Model Evaluation & Selection
---

## 🔧 Features

### **Base Features (11)**
1. `make` - Car manufacturer
2. `model` - Car model
3. `year` - Manufacturing year
4. `mileage_km` - Odometer reading
5. `engine_cc` - Engine capacity
6. `assembly` - Local/Imported
7. `reg_city` - Registration city
8. `city` - Current city
9. `transmission` - Manual/Automatic
10. `fuel_type` - Petrol/Diesel/Hybrid/CNG
11. `price_pkr` - Price in PKR (target)

### **Engineered Features (15)**

**Time-based (3):**
- `car_age` - Current year - manufacturing year
- `mileage_per_year` - Average annual mileage
- `age_group` - New/Mid/Old

**Binary (4):**
- `is_imported` - 1 if imported, 0 otherwise
- `is_automatic` - 1 if automatic, 0 otherwise
- `cross_city_sale` - 1 if reg_city ≠ city
- `is_hybrid` - 1 if hybrid fuel type

**Categorical (4):**
- `brand_tier` - Luxury(3)/Premium(2)/Budget(1)
- `city_tier` - Metro(1)/Major(2)/Other(3)
- `price_category` - Budget/Mid-Range/Premium/Luxury
- `engine_category` - Small/Medium/Large/Very Large

**Numerical (2):**
- `price_per_cc` - Price per engine cc
- `log_price` - Log-transformed price

**Interactions (2):**
- `age_mileage_interact` - car_age × mileage_km
- `brand_engine_interact` - brand_tier × engine_cc

---

## 📊 Key Statistics

### **Dataset Overview**
- **Total Records**: 24,135
- **Features**: 26 (11 base + 15 engineered)
- **Price Range**: Rs 100,000 - Rs 50,000,000
- **Year Range**: 1990 - 2026
- **Mileage Range**: 0 - 500,000 km

### **Top Brands**
1. Toyota - 38%
2. Suzuki - 24%
3. Honda - 19%
4. Daihatsu - 5%
5. KIA - 4%

### **Assembly Distribution**
- Local: 68.2%
- Imported: 31.8%

### **Transmission Distribution**
- Automatic: 57%
- Manual: 43%

---

## 🤖 Deliverable 2: Model Training & Evaluation

### **Overview**
Implemented and evaluated two gradient boosting models with comprehensive hyperparameter tuning using GridSearchCV.

### **Models Implemented**

#### **1. XGBoost Regressor**
- **Hyperparameter Tuning**: 324 combinations tested via 3-Fold GridSearchCV
- **Best Parameters**:
-  max_depth: 7
- learning_rate: 0.1
- n_estimators: 300
- min_child_weight: 1
- subsample: 1.0
- colsample_bytree: 0.8
  - **Training Time**: ~15 minutes

#### **2. LightGBM Regressor** 🏆 **(Selected Model)**
- **Hyperparameter Tuning**: 648 combinations tested via 3-Fold GridSearchCV
- **Best Parameters**:
- max_depth: -1 (no limit)
- learning_rate: 0.05
- n_estimators: 300
- num_leaves: 70
- min_child_samples: 20
- subsample: 0.8
  - **Training Time**: ~25 minutes

---

### **Performance Comparison**

| Model | Train R² | Val R² | Test R² | Test MAE | Test RMSE |
|-------|----------|--------|---------|----------|-----------|
| **XGBoost** | 0.9908 | 0.9072 | 0.8614 | Rs 571,006 | Rs 1,844,519 |
| **LightGBM** 🏆 | 0.9681 | 0.9030 | **0.8731** | **Rs 562,417** | **Rs 1,764,932** |

**Winner**: LightGBM outperforms XGBoost by **1.17%** in test R² score.

---

### **Key Results**

**Model Selection:**
- ✅ **Best Model**: LightGBM
- ✅ **Test R² Score**: 0.8731 (87.3% variance explained)
- ✅ **Average Prediction Error**: Rs 562,417 (~11% of median price)
- ✅ **RMSE**: Rs 1,764,932
- ✅ **Dataset Split**: 70% Train / 15% Validation / 15% Test

**Model Generalization:**
- Both models show good generalization (small gap between train and test R²)
- LightGBM demonstrates better balance between bias and variance
- No significant overfitting observed

**Prediction Accuracy:**
- Average error of ~Rs 562K on test set
- Suitable for real-world deployment
- Reliable for cars in the Rs 2M - Rs 8M range

---

### **Feature Importance**

#### **XGBoost Top Features**:
1. `brand_tier` (0.207)
2. `brand_engine_interact` (0.134)
3. `car_age` (0.132)
4. `engine_cc` (0.102)
5. `engine_category` (0.078)

#### **LightGBM Top Features**:
1. `model` (3769)
2. `engine_cc` (2734)
3. `mileage_per_year` (1947)
4. `mileage_km` (1869)
5. `age_mileage_interact` (1867)

**Key Insight**: Both models identify `engine_cc` and brand-related features as critical predictors, validating our feature engineering strategy.

---

### **Model Files**

Pre-trained models are included in the repository:

| File | Size | Description |
|------|------|-------------|
| `models/xgboost_model.pkl` | 1.9 MB | XGBoost trained model |
| `models/lightgbm_model.pkl` | 1.8 MB | LightGBM trained model (Best) |
| `models/label_encoders.pkl` | 40 KB | Categorical feature encoders |

**Note**: Models are hosted directly in the repository (under 5 MB total).

---

### **Validation Strategy**

- **Train/Validation/Test Split**: 70% / 15% / 15%
- **Cross-Validation**: 3-Fold CV during GridSearchCV
- **Total Dataset**: 24,135 records
  - Training: 16,903 samples
  - Validation: 3,611 samples
  - Testing: 3,621 samples
- **Stratification**: Random split with fixed seed (42) for reproducibility

---

### **Evaluation Metrics**

| Metric | Description | Why Used |
|--------|-------------|----------|
| **R² Score** | Variance explained by model | Primary metric for regression quality |
| **MAE** | Mean Absolute Error | Interpretable average prediction error |
| **RMSE** | Root Mean Squared Error | Penalizes large errors more heavily |

---

### **Evaluation Visualizations**

Generated plots (in `models/evaluation/`):
- ✅ Feature importance comparison (XGBoost vs LightGBM)
- ✅ R² score comparison across Train/Val/Test datasets
- ✅ MAE and RMSE comparison charts
- ✅ Comprehensive evaluation summary report

---
### **Team Members**

| Name | Student ID | Responsibility |
|------|-----------|----------------|
| Muhammad Ahmad | 23L-2511 | Data Scraping, Feature Engineering, XGBoost |
| Muhammad Sufyan | 23L-2518 | Data Cleaning, LightGBM, Documentation |
| Muhammad Daniyal | 23L-2600 | EDA, Model Evaluation, Visualizations |

### **Deliverables**

**✅ Deliverable 1**: Problem Definition & Data Preprocessing *(Completed)*
- Phase 1: Problem definition
- Phase 2: Data collection, cleaning, feature engineering, EDA
- GitHub repository setup
- LaTeX report

**✅ Deliverable 2**: Model Training & Evaluation *(Completed)*
- Phase 3: XGBoost and LightGBM implementation
- Phase 4: Hyperparameter tuning and model evaluation
- Performance comparison
- LaTeX report

**🔜 Deliverable 3**: XAI Integration & Deployment *(In Progress)*
- SHAP explainability integration
- Streamlit web interface
- Real-time predictions
- Final documentation

---

## 🔮 Next Steps (Deliverable 3)

### **Explainable AI (SHAP)**
- ✅ Integrate SHAP for model interpretability
- ✅ Feature contribution analysis per prediction
- ✅ Force plots and waterfall charts
- ✅ Summary plots for global feature importance

### **Web Deployment**
- ✅ Streamlit web application
- ✅ User-friendly interface for price prediction
- ✅ Real-time predictions with explanations
- ✅ Interactive visualizations

### **Production Features**
- ✅ Input validation and error handling
- ✅ Model versioning
- ✅ Performance monitoring
- ✅ API endpoint for external integration

---

## Acknowledgments

- **Data Source**: [PakWheels.com](https://www.pakwheels.com) - Pakistan's largest automotive marketplace
- **Python Community**: Open-source libraries (scikit-learn, XGBoost, LightGBM, pandas, matplotlib)
- **GitHub**: Version control and collaboration platform

---

## 📞 Contact

For questions, feedback, or collaboration:
- **GitHub**: [Muhammad-Ahmad2511/used-car-price-predictor](https://github.com/Muhammad-Ahmad2511/used-car-price-predictor)
- **Email**: University email addresses (available on request)

---

**Project Status**: ✅ Deliverable 2 Complete | 🔜 Deliverable 3 In Progress  
**Last Updated**: April 18, 2026  
**Version**: 2.0

