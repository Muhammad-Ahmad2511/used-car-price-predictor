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
- 🔜 Machine Learning models (XGBoost, LightGBM)
- 🔜 Explainable AI (SHAP integration)

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | PakWheels.com |
| **Size** |  24135 clean records |
| **Features** | 26 total (11 base + 15 engineered) |
| **Target** | Price (PKR) |
| **Time Period** | 2020-2026 |

### Sample Data

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
| **Machine Learning** | XGBoost, LightGBM (planned) |
| **Explainability** | SHAP (planned) |
| **Visualization** | Matplotlib, Seaborn |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure
```
project/
│
├── data_scripts/              # Phase 2 code (REQUIRED)
│   ├── scrape_pakwheels.py    # Web scraping
│   ├── 02_data_cleaning.py    # Data cleaning
│   └── 03_feature_engineering.py  # Feature engineering
│
├── eda/                       # EDA (REQUIRED FOR REPORT)
│   ├── eda_analysis.py        # Generate all plots
│   └── plots/                 # Generated plots
│       ├── price_distribution.png
│       ├── feature_distributions.png
│       ├── categorical_distributions.png
│       ├── correlation_matrix.png
│       └── price_vs_features.png
│
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
│
└── report/                    # LaTeX report (LATER)
    ├── deliverable1_report.tex
    └── deliverable1_report.pdf
```

---

## 🚀 Installation & Usage

### **Prerequisites**
- Python 3.11+
- Required packages: pandas, numpy, requests, beautifulsoup4, matplotlib, seaborn

### **Step 1: Install Dependencies**
```bash
pip install pandas numpy requests beautifulsoup4 matplotlib seaborn
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

---

## 📈 Data Pipeline
```
PakWheels.com → Web Scraping → Raw CSV (1,847 rows)
                     ↓
              Data Cleaning (92.2% retention)
                     ↓
              Clean CSV (1,599 rows)
                     ↓
          Feature Engineering (+15 features)
                     ↓
          Final Dataset (26 columns)
                     ↓
         EDA & Visualization (5 plots)
```

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
- **Total Records**: 1,599
- **Features**: 26 (11 base + 15 engineered)
- **Price Range**: Rs 100,000 - Rs 50,000,000
- **Year Range**: 1990 - 2026
- **Mileage Range**: 0 - 500,000 km

### **Top Brands**
1. Toyota - 35.2%
2. Suzuki - 22.8%
3. Honda - 18.5%
4. Hyundai - 7.3%
5. KIA - 5.1%

### **Assembly Distribution**
- Local: 68.4%
- Imported: 31.6%

### **Transmission Distribution**
- Automatic: 54.2%
- Manual: 45.8%

---

## 🎓 Academic Context

### **Assignment Details**
- **Course**: Artificial Intelligence
- **Track**: Track A - Application Development
- **Deliverable**: 1 of 3
- **Due Date**: March 18, 2026

### **Deliverable 1 Requirements**
- ✅ Phase 1: Problem Definition
- ✅ Phase 2: Data Collection & Preprocessing
- ✅ GitHub Repository
- ✅ LaTeX Report (PDF)

---

## 🔮 Future Work (Deliverables 2 & 3)

### **Deliverable 2: Model Training**
- Train XGBoost & LightGBM models
- Hyperparameter tuning
- Model evaluation (MAE, RMSE, R²)
- Feature importance analysis

### **Deliverable 3: Deployment**
- SHAP explainability integration
- Streamlit web interface
- Real-time price predictions
- Final documentation

---

## 👨‍💻 Group Members

**Muhammad Ahmad**  
23L-2511

**Muhammad Sufyan**  
23L-2518

**Muhammad Daniyal**  
23L-2600

AI Track A  

**GitHub**: [https://github.com/Muhammad-Ahmad2511/Desktop](https://github.com/Muhammad-Ahmad2511/Desktop)

---

## 📝 License

This project is part of an academic assignment.

---

## 🙏 Acknowledgments

- Data source: [PakWheels.com](https://www.pakwheels.com)
- Course instructors and TAs
- Python open-source community

---

## 📞 Contact

For questions or feedback, please reach out via GitHub issues or university email.

---

**Last Updated**: March 20, 2026
