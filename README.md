# 🚗 PakWheels Used Car Price Predictor

AI-powered price prediction with Explainable AI (XAI) for Pakistan's used car market.

---

## 🌐 Deployment

**Live App**: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)  
> *(Replace with your actual Streamlit Cloud URL after deployment)*

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
- ✅ Machine Learning models (XGBoost, LightGBM)
- ✅ Explainable AI (SHAP integration)
- ✅ Streamlit web application with real-time predictions

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | PakWheels.com |
| **Size** | 24,135 clean records |
| **Features** | 26 total (11 base + 15 engineered) |
| **Target** | Price (PKR) |
| **Time Period** | 2020-2026 |

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
| **Machine Learning** | XGBoost, LightGBM |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn |
| **Frontend** | Streamlit |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

```
used-car-price-predictor/
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore
│
├── data/                      # Datasets
│   ├── pakwheels_scraped.csv
│   ├── pakwheels_cleaned.csv
│   └── pakwheels_features.csv
│
├── data_scripts/              # Data pipeline
│   ├── scrape_pakwheels.py
│   ├── data_cleaning.py
│   └── feature_engineering.py
│
├── eda/                       # Exploratory Data Analysis
│   ├── eda_analysis.py
│   └── plots/
│       ├── price_distribution.png
│       ├── feature_distributions.png
│       ├── categorical_distributions.png
│       ├── correlation_matrix.png
│       └── price_vs_features.png
│
├── models/                    # Trained models & evaluation
│   ├── train_xgboost.ipynb
│   ├── train_lightgbm.ipynb
│   ├── evaluate_models.ipynb
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── label_encoders.pkl
│   ├── xgboost_results.json
│   ├── lightgbm_results.json
│   └── evaluation/
│       ├── evaluation_summary.txt
│       ├── mae_rmse_comparison.png
│       ├── r2_comparison.png
│       └── feature_importance_comparison.png
│
└── reports/                   # PDF reports
    ├── deliverable1_report.pdf
    ├── Deliverable_2_report.pdf
    └── Deliverable_3_report.pdf
```

---

## 🚀 Installation & Usage

### **Prerequisites**
- Python 3.11+

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Muhammad-Ahmad2511/used-car-price-predictor.git
cd used-car-price-predictor
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run the App**
```bash
streamlit run app.py
```

### **Step 4: (Optional) Re-run Data Pipeline**
```bash
python data_scripts/scrape_pakwheels.py
python data_scripts/data_cleaning.py
python data_scripts/feature_engineering.py
python eda/eda_analysis.py
```

---

## 📈 Data Pipeline

```
PakWheels.com → Web Scraping → Raw CSV
                     ↓
              Data Cleaning (92.2% retention)
                     ↓
              Clean CSV (24,135 rows)
                     ↓
          Feature Engineering (+15 features)
                     ↓
          Final Dataset (26 columns)
                     ↓
         EDA & Visualization (5 plots)
                     ↓
        Model Training (XGBoost & LightGBM)
                     ↓
        SHAP Explainability + Streamlit App
```

---

## 📊 Model Performance

| Metric | XGBoost | LightGBM |
|--------|---------|----------|
| **R² (Test)** | 0.861 | **0.873** ✅ |
| **MAE (Test)** | Rs 571,006 | **Rs 562,417** ✅ |
| **RMSE (Test)** | Rs 1,844,519 | Rs 1,764,932 |

**Best Model: LightGBM** (+1.17% R² vs XGBoost)

### Top Features (LightGBM)
1. `model` — Car model name
2. `engine_cc` — Engine capacity
3. `mileage_per_year` — Average annual mileage

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

**Time-based (3):** `car_age`, `mileage_per_year`, `age_group`  
**Binary (4):** `is_imported`, `is_automatic`, `cross_city_sale`, `is_hybrid`  
**Categorical (4):** `brand_tier`, `city_tier`, `price_category`, `engine_category`  
**Numerical (2):** `price_per_cc`, `log_price`  
**Interactions (2):** `age_mileage_interact`, `brand_engine_interact`  

---

## 📊 Key Statistics

- **Total Records**: 24,135
- **Price Range**: Rs 100,000 – Rs 50,000,000
- **Year Range**: 1990 – 2026

### Top Brands
1. Toyota – 35.2%
2. Suzuki – 22.8%
3. Honda – 18.5%
4. Hyundai – 7.3%
5. KIA – 5.1%

---

## 🎓 Academic Context

- **Course**: Artificial Intelligence
- **Track**: Track A — Application Development
- **Deliverables**: 3 of 3 ✅

---

## 👨‍💻 Group Members

**Muhammad Ahmad** — 23L-2511  
**Muhammad Sufyan** — 23L-2518  
**Muhammad Daniyal** — 23L-2600  

AI Track A

**GitHub**: [https://github.com/Muhammad-Ahmad2511/used-car-price-predictor](https://github.com/Muhammad-Ahmad2511/used-car-price-predictor)

---

## 🙏 Acknowledgments

- Data source: [PakWheels.com](https://www.pakwheels.com)
- Course instructors and TAs
- Python open-source community

---

**Last Updated**: May 2026
