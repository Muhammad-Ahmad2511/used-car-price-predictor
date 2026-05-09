import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PakWheels Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS  (cached — runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    lgbm_model     = joblib.load("models/lightgbm_model.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    lgbm_explainer = shap.TreeExplainer(lgbm_model)
    return lgbm_model, label_encoders, lgbm_explainer

lgbm_model, label_encoders, lgbm_explainer = load_artifacts()

# ─────────────────────────────────────────────
# FEATURE ORDER
# Both models were trained on the same 22 columns in this exact order.
# Both predict raw price_pkr (no log transform / expm1 needed).
# ─────────────────────────────────────────────
FEATURES = [
    'make', 'model', 'year', 'mileage_km', 'engine_cc',
    'assembly', 'reg_city', 'city', 'transmission', 'fuel_type',
    'car_age', 'mileage_per_year', 'age_group', 'is_imported',
    'is_automatic', 'cross_city_sale', 'is_hybrid', 'brand_tier',
    'city_tier', 'engine_category', 'age_mileage_interact', 'brand_engine_interact'
]

# ─────────────────────────────────────────────
# MODEL METADATA
# ─────────────────────────────────────────────
MODEL_INFO = {
    "LightGBM": {"r2": 0.873, "mae_val": 562_000, "model": lgbm_model, "explainer": lgbm_explainer},
}
MAE = 562_000  # LightGBM test MAE in Rs

# ─────────────────────────────────────────────
# OPTION LISTS
# ─────────────────────────────────────────────
MAKE_MODELS = {
    "Toyota":       ["Corolla", "Corolla Gli 1.3 Vvti", "Corolla Gli Automatic 1.6 Vvti",
                     "Corolla Altis 1.8", "Corolla Altis Grande 1.8", "Corolla Xli Vvti",
                     "Prius", "Prius S 1.8", "Prius A", "Aqua", "Aqua G",
                     "Vitz", "Vitz F 1.0", "Vitz F 1.3",
                     "Fortuner", "Fortuner 2.7 V", "Fortuner Legender",
                     "Prado", "Prado Tx 2.7", "Prado Tz G 4.0",
                     "Land Cruiser", "Land Cruiser Zx", "Land Cruiser Vx 4.7",
                     "Hilux", "Hilux Revo V Automatic 2.8", "Hilux Revo G Automatic 2.8",
                     "Other"],
    "Honda":        ["Civic", "Civic Oriel", "Civic Vti Oriel 1.8 I-Vtec", "Civic Rs",
                     "Civic 1.5 Rs Turbo", "City", "City 1.3 I-Vtec Prosmatec",
                     "City 1.5L Cvt", "City 1.5L Aspire Cvt",
                     "Vezel", "Vezel Hybrid X", "Vezel E-Hev Z",
                     "Hr-V", "Cr-V", "Br-V", "Br-V I-Vtec", "Other"],
    "Suzuki":       ["Alto", "Alto Vxr", "Alto Vxl Ags", "Alto Vxr (Cng)",
                     "Wagon R", "Wagon R Vxr", "Wagon R Vxl", "Wagon R Ags",
                     "Cultus", "Cultus Vxl", "Cultus Vxr", "Cultus Vxli",
                     "Swift", "Swift Dlx 1.3", "Swift Gl Cvt",
                     "Mehran Vx", "Mehran Vxr", "Bolan Vx", "Other"],
    "Kia":          ["Sportage", "Sportage Alpha", "Sportage Fwd", "Sportage Awd",
                     "Picanto", "Carnival", "Other"],
    "Hyundai":      ["Tucson", "Tucson Fwd A/T Gls", "Tucson Hybrid Smart", "Other"],
    "Daihatsu":     ["Mira", "Mira G Sa Iii", "Mira X", "Cuore", "Cuore Cx", "Other"],
    "Nissan":       ["Other"],
    "Mitsubishi":   ["Pajero", "Pajero Exceed 3.5", "Lancer Glx 1.5", "Lancer Glx 1.6", "Other"],
    "Mg":           ["Other"],
    "Haval":        ["Other"],
    "Changan":      ["Other"],
    "Chery":        ["Other"],
    "Audi":         ["A3", "A4", "A6", "A8", "Q5", "Q7", "Q8", "Other"],
    "Bmw":          ["3 Series", "5 Series", "7 Series", "X3", "X5", "Other"],
    "Mercedes":     ["C-Class", "E-Class", "S-Class", "GLC", "GLE", "Other"],
    "Mercedes Benz":["C-Class", "E-Class", "S-Class", "GLC", "GLE", "Other"],
    "Lexus":        ["Es 350", "Lx 570", "Rx 350", "Other"],
    "Volkswagen":   ["Golf", "Passat", "Tiguan", "Other"],
    "Mazda":        ["Other"],
    "Subaru":       ["Other"],
    "Ford":         ["Other"],
    "Jeep":         ["Other"],
    "Proton":       ["Other"],
    "Faw":          ["Other"],
    "Dfsk":         ["Other"],
    "Jac":          ["Other"],
    "Byd":          ["Other"],
    "Tesla":        ["Other"],
}

# Earliest valid model year per car model (Pakistan market / global launch).
# Falls back to MAKE_MIN_YEAR if model not listed, then 1990.
MODEL_MIN_YEAR = {
    # Toyota
    "Corolla": 1966, "Corolla Gli 1.3 Vvti": 2002,
    "Corolla Gli Automatic 1.6 Vvti": 2009, "Corolla Altis 1.8": 2014,
    "Corolla Altis Grande 1.8": 2016, "Corolla Xli Vvti": 2002,
    "Prius": 1997, "Prius S 1.8": 2012, "Prius A": 2012,
    "Aqua": 2012, "Aqua G": 2012,
    "Vitz": 1999, "Vitz F 1.0": 2011, "Vitz F 1.3": 2011,
    "Fortuner": 2005, "Fortuner 2.7 V": 2005, "Fortuner Legender": 2020,
    "Prado": 1996, "Prado Tx 2.7": 2003, "Prado Tz G 4.0": 2003,
    "Land Cruiser": 1990, "Land Cruiser Zx": 2021, "Land Cruiser Vx 4.7": 1998,
    "Hilux": 1990, "Hilux Revo V Automatic 2.8": 2016, "Hilux Revo G Automatic 2.8": 2016,
    # Honda
    "Civic": 1994, "Civic Oriel": 2004, "Civic Vti Oriel 1.8 I-Vtec": 2006,
    "Civic Rs": 2022, "Civic 1.5 Rs Turbo": 2017,
    "City": 1996, "City 1.3 I-Vtec Prosmatec": 2009,
    "City 1.5L Cvt": 2021, "City 1.5L Aspire Cvt": 2021,
    "Vezel": 2014, "Vezel Hybrid X": 2014, "Vezel E-Hev Z": 2022,
    "Hr-V": 2015, "Cr-V": 1997, "Br-V": 2017, "Br-V I-Vtec": 2017,
    # Suzuki
    "Alto": 2000, "Alto Vxr": 2000, "Alto Vxl Ags": 2019, "Alto Vxr (Cng)": 2000,
    "Wagon R": 2014, "Wagon R Vxr": 2014, "Wagon R Vxl": 2014, "Wagon R Ags": 2018,
    "Cultus": 2000, "Cultus Vxl": 2017, "Cultus Vxr": 2017, "Cultus Vxli": 2000,
    "Swift": 2010, "Swift Dlx 1.3": 2010, "Swift Gl Cvt": 2022,
    "Mehran Vx": 1990, "Mehran Vxr": 1990, "Bolan Vx": 1990,
    # Kia
    "Sportage": 2019, "Sportage Alpha": 2019, "Sportage Fwd": 2019, "Sportage Awd": 2019,
    "Picanto": 2019, "Carnival": 2021,
    # Hyundai
    "Tucson": 2021, "Tucson Fwd A/T Gls": 2021, "Tucson Hybrid Smart": 2022,
    # Daihatsu
    "Mira": 2006, "Mira G Sa Iii": 2017, "Mira X": 2006,
    "Cuore": 2000, "Cuore Cx": 2000,
    # Mitsubishi
    "Pajero": 1997, "Pajero Exceed 3.5": 1997,
    "Lancer Glx 1.5": 2004, "Lancer Glx 1.6": 2004,
    # Audi
    "A3": 2014, "A4": 2006, "A6": 2005, "A8": 2010,
    "Q5": 2017, "Q7": 2016, "Q8": 2020,
    # BMW
    "3 Series": 2000, "5 Series": 2000, "7 Series": 2000,
    "X3": 2010, "X5": 2006, "X6": 2012,
    # Mercedes / Mercedes Benz
    "C-Class": 2000, "E-Class": 2000, "S-Class": 2000, "GLC": 2016, "GLE": 2016,
    # Lexus
    "Es 350": 2010, "Lx 570": 2012, "Rx 350": 2010,
    # Volkswagen
    "Golf": 2010, "Passat": 2010, "Tiguan": 2018,
    # Default
    "Other": 1990,
}

MAKE_MIN_YEAR = {
    "Toyota": 1990, "Honda": 1996, "Suzuki": 1990, "Kia": 2019,
    "Hyundai": 2021, "Daihatsu": 2000, "Nissan": 1990, "Mitsubishi": 1990,
    "Mg": 2020, "Haval": 2020, "Changan": 2018, "Chery": 2020,
    "Audi": 2004, "Bmw": 2000, "Mercedes": 2000, "Mercedes Benz": 2000,
    "Lexus": 2008, "Volkswagen": 2010, "Mazda": 2010, "Subaru": 2010,
    "Ford": 2010, "Jeep": 2010, "Proton": 2021, "Faw": 2016,
    "Dfsk": 2019, "Jac": 2019, "Byd": 2022, "Tesla": 2023, "Other": 1990,
}

# Valid fuel types per model. Falls back to MAKE_FUEL_TYPES if model not listed.
MODEL_FUEL_TYPES = {
    # Toyota — hybrids
    "Prius": ["Hybrid"], "Prius S 1.8": ["Hybrid"], "Prius A": ["Hybrid"],
    "Aqua": ["Hybrid"], "Aqua G": ["Hybrid"],
    "Fortuner Legender": ["Petrol", "Diesel"],
    "Fortuner": ["Petrol", "Diesel"], "Fortuner 2.7 V": ["Petrol"],
    "Hilux": ["Diesel"], "Hilux Revo V Automatic 2.8": ["Diesel"], "Hilux Revo G Automatic 2.8": ["Diesel"],
    "Prado": ["Petrol", "Diesel"], "Prado Tx 2.7": ["Petrol"], "Prado Tz G 4.0": ["Petrol"],
    "Land Cruiser": ["Petrol", "Diesel"], "Land Cruiser Zx": ["Petrol"], "Land Cruiser Vx 4.7": ["Petrol"],
    "Corolla": ["Petrol"], "Corolla Gli 1.3 Vvti": ["Petrol"], "Corolla Gli Automatic 1.6 Vvti": ["Petrol"],
    "Corolla Altis 1.8": ["Petrol"], "Corolla Altis Grande 1.8": ["Petrol"], "Corolla Xli Vvti": ["Petrol"],
    "Vitz": ["Petrol"], "Vitz F 1.0": ["Petrol"], "Vitz F 1.3": ["Petrol"],
    # Honda
    "Civic": ["Petrol"], "Civic Oriel": ["Petrol"], "Civic Vti Oriel 1.8 I-Vtec": ["Petrol"],
    "Civic Rs": ["Petrol"], "Civic 1.5 Rs Turbo": ["Petrol"],
    "City": ["Petrol"], "City 1.3 I-Vtec Prosmatec": ["Petrol"],
    "City 1.5L Cvt": ["Petrol"], "City 1.5L Aspire Cvt": ["Petrol"],
    "Vezel": ["Hybrid"], "Vezel Hybrid X": ["Hybrid"], "Vezel E-Hev Z": ["Hybrid"],
    "Hr-V": ["Petrol"], "Cr-V": ["Petrol"], "Br-V": ["Petrol"], "Br-V I-Vtec": ["Petrol"],
    # Suzuki — Alto has CNG variant too
    "Alto": ["Petrol", "CNG"], "Alto Vxr": ["Petrol", "CNG"],
    "Alto Vxl Ags": ["Petrol"], "Alto Vxr (Cng)": ["CNG"],
    "Wagon R": ["Petrol", "CNG"], "Wagon R Vxr": ["Petrol", "CNG"],
    "Wagon R Vxl": ["Petrol"], "Wagon R Ags": ["Petrol"],
    "Cultus": ["Petrol"], "Cultus Vxl": ["Petrol"], "Cultus Vxr": ["Petrol"], "Cultus Vxli": ["Petrol"],
    "Swift": ["Petrol"], "Swift Dlx 1.3": ["Petrol"], "Swift Gl Cvt": ["Petrol"],
    "Mehran Vx": ["Petrol", "CNG"], "Mehran Vxr": ["Petrol", "CNG"], "Bolan Vx": ["Petrol", "CNG"],
    # Kia
    "Sportage": ["Petrol"], "Sportage Alpha": ["Petrol"],
    "Sportage Fwd": ["Petrol"], "Sportage Awd": ["Petrol"],
    "Picanto": ["Petrol"], "Carnival": ["Petrol"],
    # Hyundai
    "Tucson": ["Petrol"], "Tucson Fwd A/T Gls": ["Petrol"],
    "Tucson Hybrid Smart": ["Hybrid"],
    # Daihatsu
    "Mira": ["Petrol"], "Mira G Sa Iii": ["Petrol"], "Mira X": ["Petrol"],
    "Cuore": ["Petrol"], "Cuore Cx": ["Petrol"],
    # Mitsubishi
    "Pajero": ["Petrol", "Diesel"], "Pajero Exceed 3.5": ["Petrol"],
    "Lancer Glx 1.5": ["Petrol"], "Lancer Glx 1.6": ["Petrol"],
    # Audi
    "A3": ["Petrol"], "A4": ["Petrol", "Diesel"], "A6": ["Petrol", "Diesel"],
    "A8": ["Petrol"], "Q5": ["Petrol"], "Q7": ["Petrol"], "Q8": ["Petrol"],
    # BMW
    "3 Series": ["Petrol", "Diesel"], "5 Series": ["Petrol", "Diesel"],
    "7 Series": ["Petrol"], "X3": ["Petrol", "Diesel"],
    "X5": ["Petrol", "Diesel"], "X6": ["Petrol"],
    # Mercedes
    "C-Class": ["Petrol", "Diesel"], "E-Class": ["Petrol", "Diesel"],
    "S-Class": ["Petrol"], "GLC": ["Petrol"], "GLE": ["Petrol", "Diesel"],
    # Lexus
    "Es 350": ["Hybrid"], "Lx 570": ["Petrol"], "Rx 350": ["Petrol"],
    # Volkswagen
    "Golf": ["Petrol"], "Passat": ["Petrol", "Diesel"], "Tiguan": ["Petrol"],
    # Default
    "Other": ["Petrol", "Hybrid", "Diesel", "CNG", "Electric", "Lpg", "Phev"],
}

MAKE_FUEL_TYPES = {
    "Toyota": ["Petrol", "Hybrid", "Diesel"],
    "Honda": ["Petrol", "Hybrid"],
    "Suzuki": ["Petrol", "CNG"],
    "Kia": ["Petrol"],
    "Hyundai": ["Petrol", "Hybrid"],
    "Daihatsu": ["Petrol"],
    "Nissan": ["Petrol", "Diesel"],
    "Mitsubishi": ["Petrol", "Diesel"],
    "Mg": ["Petrol", "Electric"],
    "Haval": ["Petrol", "Hybrid"],
    "Changan": ["Petrol"],
    "Chery": ["Petrol"],
    "Audi": ["Petrol", "Diesel"],
    "Bmw": ["Petrol", "Diesel"],
    "Mercedes": ["Petrol", "Diesel"],
    "Mercedes Benz": ["Petrol", "Diesel"],
    "Lexus": ["Petrol", "Hybrid"],
    "Volkswagen": ["Petrol", "Diesel"],
    "Mazda": ["Petrol"],
    "Subaru": ["Petrol"],
    "Ford": ["Petrol", "Diesel"],
    "Jeep": ["Petrol"],
    "Proton": ["Petrol"],
    "Faw": ["Petrol", "Diesel"],
    "Dfsk": ["Petrol", "Diesel"],
    "Jac": ["Petrol", "Electric"],
    "Byd": ["Electric"],
    "Tesla": ["Electric"],
    "Other": ["Petrol", "Hybrid", "Diesel", "CNG", "Electric", "Lpg", "Phev"],
}

# Valid transmissions per model. Falls back to MAKE_TRANSMISSIONS if model not listed.
MODEL_TRANSMISSIONS = {
    # Toyota
    "Corolla": ["Automatic", "Manual"],
    "Corolla Gli 1.3 Vvti": ["Manual"],
    "Corolla Gli Automatic 1.6 Vvti": ["Automatic"],
    "Corolla Altis 1.8": ["Automatic", "Manual"],
    "Corolla Altis Grande 1.8": ["Automatic"],
    "Corolla Xli Vvti": ["Manual"],
    "Prius": ["Automatic"], "Prius S 1.8": ["Automatic"], "Prius A": ["Automatic"],
    "Aqua": ["Automatic"], "Aqua G": ["Automatic"],
    "Vitz": ["Automatic", "Manual"], "Vitz F 1.0": ["Manual"], "Vitz F 1.3": ["Automatic"],
    "Fortuner": ["Automatic", "Manual"], "Fortuner 2.7 V": ["Automatic"], "Fortuner Legender": ["Automatic"],
    "Prado": ["Automatic"], "Prado Tx 2.7": ["Automatic"], "Prado Tz G 4.0": ["Automatic"],
    "Land Cruiser": ["Automatic"], "Land Cruiser Zx": ["Automatic"], "Land Cruiser Vx 4.7": ["Automatic"],
    "Hilux": ["Automatic", "Manual"],
    "Hilux Revo V Automatic 2.8": ["Automatic"], "Hilux Revo G Automatic 2.8": ["Automatic"],
    # Honda
    "Civic": ["Automatic", "Manual"], "Civic Oriel": ["Automatic", "Manual"],
    "Civic Vti Oriel 1.8 I-Vtec": ["Automatic"],
    "Civic Rs": ["Automatic"], "Civic 1.5 Rs Turbo": ["Automatic"],
    "City": ["Automatic", "Manual"], "City 1.3 I-Vtec Prosmatec": ["Automatic"],
    "City 1.5L Cvt": ["Automatic"], "City 1.5L Aspire Cvt": ["Automatic"],
    "Vezel": ["Automatic"], "Vezel Hybrid X": ["Automatic"], "Vezel E-Hev Z": ["Automatic"],
    "Hr-V": ["Automatic"], "Cr-V": ["Automatic"],
    "Br-V": ["Automatic", "Manual"], "Br-V I-Vtec": ["Manual"],
    # Suzuki
    "Alto": ["Automatic", "Manual"], "Alto Vxr": ["Manual"],
    "Alto Vxl Ags": ["Automatic"], "Alto Vxr (Cng)": ["Manual"],
    "Wagon R": ["Automatic", "Manual"], "Wagon R Vxr": ["Manual"],
    "Wagon R Vxl": ["Manual"], "Wagon R Ags": ["Automatic"],
    "Cultus": ["Automatic", "Manual"], "Cultus Vxl": ["Manual"],
    "Cultus Vxr": ["Manual"], "Cultus Vxli": ["Manual"],
    "Swift": ["Automatic", "Manual"], "Swift Dlx 1.3": ["Manual"], "Swift Gl Cvt": ["Automatic"],
    "Mehran Vx": ["Manual"], "Mehran Vxr": ["Manual"], "Bolan Vx": ["Manual"],
    # Kia
    "Sportage": ["Automatic"], "Sportage Alpha": ["Automatic"],
    "Sportage Fwd": ["Automatic"], "Sportage Awd": ["Automatic"],
    "Picanto": ["Automatic", "Manual"], "Carnival": ["Automatic"],
    # Hyundai
    "Tucson": ["Automatic"], "Tucson Fwd A/T Gls": ["Automatic"], "Tucson Hybrid Smart": ["Automatic"],
    # Daihatsu
    "Mira": ["Automatic"], "Mira G Sa Iii": ["Automatic"], "Mira X": ["Automatic"],
    "Cuore": ["Automatic", "Manual"], "Cuore Cx": ["Manual"],
    # Mitsubishi
    "Pajero": ["Automatic"], "Pajero Exceed 3.5": ["Automatic"],
    "Lancer Glx 1.5": ["Automatic", "Manual"], "Lancer Glx 1.6": ["Automatic"],
    # Audi
    "A3": ["Automatic"], "A4": ["Automatic"], "A6": ["Automatic"],
    "A8": ["Automatic"], "Q5": ["Automatic"], "Q7": ["Automatic"], "Q8": ["Automatic"],
    # BMW
    "3 Series": ["Automatic"], "5 Series": ["Automatic"], "7 Series": ["Automatic"],
    "X3": ["Automatic"], "X5": ["Automatic"], "X6": ["Automatic"],
    # Mercedes
    "C-Class": ["Automatic"], "E-Class": ["Automatic"],
    "S-Class": ["Automatic"], "GLC": ["Automatic"], "GLE": ["Automatic"],
    # Lexus
    "Es 350": ["Automatic"], "Lx 570": ["Automatic"], "Rx 350": ["Automatic"],
    # Volkswagen
    "Golf": ["Automatic", "Manual"], "Passat": ["Automatic"], "Tiguan": ["Automatic"],
    # Default
    "Other": ["Automatic", "Manual"],
}

MAKE_TRANSMISSIONS = {
    "Toyota": ["Automatic", "Manual"], "Honda": ["Automatic", "Manual"],
    "Suzuki": ["Automatic", "Manual"], "Kia": ["Automatic"],
    "Hyundai": ["Automatic"], "Daihatsu": ["Automatic"],
    "Nissan": ["Automatic", "Manual"], "Mitsubishi": ["Automatic", "Manual"],
    "Mg": ["Automatic"], "Haval": ["Automatic"], "Changan": ["Automatic"], "Chery": ["Automatic"],
    "Audi": ["Automatic"], "Bmw": ["Automatic"], "Mercedes": ["Automatic"],
    "Mercedes Benz": ["Automatic"], "Lexus": ["Automatic"],
    "Volkswagen": ["Automatic", "Manual"], "Mazda": ["Automatic"], "Subaru": ["Automatic"],
    "Ford": ["Automatic", "Manual"], "Jeep": ["Automatic"], "Proton": ["Automatic"],
    "Faw": ["Automatic", "Manual"], "Dfsk": ["Automatic", "Manual"],
    "Jac": ["Automatic"], "Byd": ["Automatic"], "Tesla": ["Automatic"],
    "Other": ["Automatic", "Manual"],
}

CITIES = [
    "Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad",
    "Multan", "Peshawar", "Quetta", "Sialkot", "Gujranwala",
    "Hyderabad", "Gujrat", "Sargodha", "Bahawalpur"
]

TRANSMISSIONS = ["Automatic", "Manual"]
ASSEMBLIES    = ["Local", "Imported"]
FUEL_TYPES    = ["Petrol", "Hybrid", "Diesel", "CNG", "Electric", "Lpg", "Phev"]

# Engine cc range (min, max) per model. Falls back to (600, 5000) if not listed.
MODEL_ENGINE_RANGE = {
    "Alto": (660, 660), "Alto Vxr": (660, 660), "Alto Vxl Ags": (660, 660), "Alto Vxr (Cng)": (660, 660),
    "Mehran Vx": (800, 800), "Mehran Vxr": (800, 800),
    "Wagon R": (1000, 1000), "Wagon R Vxr": (1000, 1000), "Wagon R Vxl": (1000, 1000), "Wagon R Ags": (1000, 1000),
    "Cultus": (1000, 1300), "Cultus Vxl": (1000, 1000), "Cultus Vxr": (1000, 1000), "Cultus Vxli": (1300, 1300),
    "Bolan Vx": (800, 800),
    "Swift": (1200, 1300), "Swift Dlx 1.3": (1300, 1300), "Swift Gl Cvt": (1200, 1200),
    "Corolla": (1300, 1800), "Corolla Gli 1.3 Vvti": (1300, 1300),
    "Corolla Gli Automatic 1.6 Vvti": (1600, 1600), "Corolla Xli Vvti": (1300, 1300),
    "Corolla Altis 1.8": (1800, 1800), "Corolla Altis Grande 1.8": (1800, 1800),
    "Civic": (1500, 1800), "Civic Oriel": (1800, 1800), "Civic Vti Oriel 1.8 I-Vtec": (1800, 1800),
    "Civic Rs": (1500, 1500), "Civic 1.5 Rs Turbo": (1500, 1500),
    "City": (1300, 1500), "City 1.3 I-Vtec Prosmatec": (1300, 1300),
    "City 1.5L Cvt": (1500, 1500), "City 1.5L Aspire Cvt": (1500, 1500),
    "Prius": (1800, 1800), "Prius S 1.8": (1800, 1800), "Prius A": (1800, 1800),
    "Aqua": (1500, 1500), "Aqua G": (1500, 1500),
    "Vitz": (1000, 1300), "Vitz F 1.0": (1000, 1000), "Vitz F 1.3": (1300, 1300),
    "Vezel": (1500, 1500), "Vezel Hybrid X": (1500, 1500), "Vezel E-Hev Z": (1500, 1500),
    "Hr-V": (1800, 1800), "Cr-V": (2000, 2400), "Br-V": (1500, 1500), "Br-V I-Vtec": (1500, 1500),
    "Fortuner": (2700, 2800), "Fortuner 2.7 V": (2700, 2700), "Fortuner Legender": (2800, 2800),
    "Hilux": (2800, 2800), "Hilux Revo V Automatic 2.8": (2800, 2800), "Hilux Revo G Automatic 2.8": (2800, 2800),
    "Prado": (2700, 4000), "Prado Tx 2.7": (2700, 2700), "Prado Tz G 4.0": (4000, 4000),
    "Land Cruiser": (4000, 4700), "Land Cruiser Zx": (3300, 3300), "Land Cruiser Vx 4.7": (4700, 4700),
    "Sportage": (2000, 2000), "Sportage Alpha": (2000, 2000), "Sportage Fwd": (2000, 2000), "Sportage Awd": (2000, 2000),
    "Picanto": (1000, 1200), "Carnival": (3500, 3500),
    "Tucson": (2000, 2000), "Tucson Fwd A/T Gls": (2000, 2000), "Tucson Hybrid Smart": (1600, 1600),
    "Mira": (660, 660), "Mira G Sa Iii": (660, 660), "Mira X": (660, 660),
    "Cuore": (850, 850), "Cuore Cx": (850, 850),
    "Pajero": (3000, 3500), "Pajero Exceed 3.5": (3500, 3500),
    "Lancer Glx 1.5": (1500, 1500), "Lancer Glx 1.6": (1600, 1600),
    # Audi
    "A3": (1400, 2000), "A4": (1400, 3000), "A6": (1984, 3000),
    "A8": (3000, 4200), "Q5": (1984, 3000), "Q7": (2995, 3600), "Q8": (2995, 4000),
    # BMW
    "3 Series": (1998, 3000), "5 Series": (1998, 3000), "7 Series": (2998, 4400),
    "X3": (1998, 3000), "X5": (2998, 4400), "X6": (2998, 4400),
    # Mercedes / Mercedes Benz
    "C-Class": (1500, 3000), "E-Class": (1991, 3500), "S-Class": (2996, 5500),
    "GLC": (1991, 2996), "GLE": (1991, 2996),
    # Lexus
    "Es 350": (3456, 3456), "Lx 570": (5663, 5663), "Rx 350": (3456, 3456),
    # Volkswagen
    "Golf": (1200, 2000), "Passat": (1400, 2000), "Tiguan": (1400, 2000),
    "Other": (600, 6000),
}

# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors feature_engineering.py)
# ─────────────────────────────────────────────
LUXURY_BRANDS  = ['Mercedes', 'Mercedes Benz', 'Bmw', 'Audi', 'Porsche', 'Lexus',
                  'Jaguar', 'Land Rover', 'Range Rover', 'Bentley']
PREMIUM_BRANDS = ['Toyota', 'Honda', 'Hyundai', 'Kia', 'Nissan', 'Mazda',
                  'Volkswagen', 'Mitsubishi', 'Changan', 'Mg', 'Haval']
METRO_CITIES   = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi']
MAJOR_CITIES   = ['Faisalabad', 'Multan', 'Peshawar', 'Quetta',
                  'Sialkot', 'Gujranwala', 'Hyderabad']

def get_brand_tier(make: str) -> int:
    m = str(make).strip().title()
    if m in [str(lb).title() for lb in LUXURY_BRANDS]:  return 3
    if m in [str(pb).title() for pb in PREMIUM_BRANDS]: return 2
    return 1

def get_age_group(age: int) -> str:
    if age <= 3: return 'New'
    if age <= 7: return 'Mid'
    return 'Old'

def get_engine_category(cc: int) -> str:
    if cc < 1000:  return 'Small'
    if cc < 1600:  return 'Medium'
    if cc < 2500:  return 'Large'
    return 'Very Large'

def get_city_tier(city: str) -> int:
    c = str(city).title()
    if c in METRO_CITIES: return 1
    if c in MAJOR_CITIES: return 2
    return 3

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def safe_encode(le, val: str, field: str = "") -> int:
    if val in le.classes_:
        return int(le.transform([val])[0])
    # Value not in encoder (e.g. newly added models not in training data).
    # Try to fall back to "Other" if it exists in the encoder, else use index 0.
    # No warning shown — this is expected for models added after training.
    if "Other" in le.classes_:
        return int(le.transform(["Other"])[0])
    return 0

def build_input_df(make, car_model, year, mileage_km, engine_cc,
                   transmission, assembly, fuel_type, city, reg_city) -> pd.DataFrame:
    current_year     = datetime.date.today().year          # dynamic — never goes stale
    car_age          = max(current_year - year, 0)         # clamp to 0 for current-year cars
    mileage_per_year = int(mileage_km / max(car_age, 1))
    age_mileage_int  = car_age * mileage_km
    bt               = get_brand_tier(make)
    brand_engine_int = bt * engine_cc
    is_imported      = 1 if assembly == "Imported"               else 0
    is_automatic     = 1 if transmission == "Automatic"          else 0
    is_hybrid        = 1 if fuel_type in ("Hybrid", "Phev")      else 0
    cross_city_sale  = 1 if city != reg_city                     else 0
    city_tier        = get_city_tier(city)

    le = label_encoders
    row = {
        'make':                  safe_encode(le['make'],            make,         'make'),
        'model':                 safe_encode(le['model'],           car_model,    'model'),
        'year':                  year,
        'mileage_km':            mileage_km,
        'engine_cc':             engine_cc,
        'assembly':              safe_encode(le['assembly'],        assembly,     'assembly'),
        'reg_city':              safe_encode(le['reg_city'],        reg_city,     'reg_city'),
        'city':                  safe_encode(le['city'],            city,         'city'),
        'transmission':          safe_encode(le['transmission'],    transmission, 'transmission'),
        'fuel_type':             safe_encode(le['fuel_type'],       fuel_type,    'fuel_type'),
        'car_age':               car_age,
        'mileage_per_year':      mileage_per_year,
        'age_group':             safe_encode(le['age_group'],       get_age_group(car_age), 'age_group'),
        'is_imported':           is_imported,
        'is_automatic':          is_automatic,
        'cross_city_sale':       cross_city_sale,
        'is_hybrid':             is_hybrid,
        'brand_tier':            bt,
        'city_tier':             city_tier,
        'engine_category':       safe_encode(le['engine_category'], get_engine_category(engine_cc), 'engine_category'),
        'age_mileage_interact':  age_mileage_int,
        'brand_engine_interact': brand_engine_int,
    }
    # Enforce exact column order that both models were trained on
    return pd.DataFrame([row])[FEATURES]

# ─────────────────────────────────────────────
# READABLE LABELS
# ─────────────────────────────────────────────
FEATURE_LABELS = {
    'make':                  'Make',
    'model':                 'Model',
    'year':                  'Year',
    'mileage_km':            'Total Mileage (km)',
    'engine_cc':             'Engine (cc)',
    'assembly':              'Assembly',
    'reg_city':              'Reg. City',
    'city':                  'Listing City',
    'transmission':          'Transmission',
    'fuel_type':             'Fuel Type',
    'car_age':               'Car Age (yrs)',
    'mileage_per_year':      'Avg km/year',
    'age_group':             'Age Group',
    'is_imported':           'Imported',
    'is_automatic':          'Automatic',
    'cross_city_sale':       'Cross-City Sale',
    'is_hybrid':             'Hybrid',
    'brand_tier':            'Brand Tier',
    'city_tier':             'City Tier',
    'engine_category':       'Engine Category',
    'age_mileage_interact':  'Age x Mileage',
    'brand_engine_interact': 'Brand x Engine',
}

READABLE_NAMES = [FEATURE_LABELS.get(f, f) for f in FEATURES]

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Stat cards in header ── */
.stat-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    color: #e2e8f0;
}
.stat-card .label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
.stat-card .value { font-size: 22px; font-weight: 700; color: #38bdf8; margin-top: 2px; }

/* ── Section headers ── */
.section-header {
    font-size: 13px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 18px 0 10px 0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
}

/* ── Price result box ── */
.price-box {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    color: white;
    margin: 16px 0;
}
.price-box .price-label { font-size: 13px; opacity: 0.85; letter-spacing: 1px; text-transform: uppercase; }
.price-box .price-main  { font-size: 42px; font-weight: 800; margin: 6px 0; letter-spacing: -1px; }
.price-box .price-range { font-size: 14px; opacity: 0.80; margin-top: 4px; }
.price-box .price-note  { font-size: 12px; opacity: 0.65; margin-top: 8px; }

/* ── Factor cards ── */
.factor-card {
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.factor-card.up   { background: #f0fdf4; border-left: 5px solid #16a34a; }
.factor-card.down { background: #fff1f2; border-left: 5px solid #dc2626; }
.factor-card .feat-name  { font-weight: 600; font-size: 15px; color: #1e293b; }
.factor-card .feat-desc  { font-size: 12px; color: #64748b; margin-top: 2px; }
.factor-card .feat-val   { font-weight: 700; font-size: 16px; white-space: nowrap; }
.factor-card.up   .feat-val { color: #16a34a; }
.factor-card.down .feat-val { color: #dc2626; }

/* ── Explainer box ── */
.explain-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 14px;
    color: #334155;
    line-height: 1.7;
    margin-bottom: 16px;
}
.explain-box strong { color: #0f172a; }

/* ── How it works steps ── */
.how-step {
    background: #f8fafc;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #475569;
    border-left: 3px solid #38bdf8;
}
.how-step b { color: #0f172a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.markdown("## 🚗 PakWheels Car Price Predictor")
st.markdown(
    "Get an **AI-powered resale price estimate** for any car listed on PakWheels — "
    "powered by a LightGBM model trained on thousands of real listings."
)

st.divider()

# Hardcode best model (LightGBM, R2=0.873)
info = MODEL_INFO["LightGBM"]


# ─────────────────────────────────────────────
# UI — INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔧 Car Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Identity**")
    _makes_raw = list(MAKE_MODELS.keys())
    _makes_sorted = sorted(m for m in _makes_raw if m != "Other")
    make      = st.selectbox("Make (Brand)", _makes_sorted,
                              help="The manufacturer — e.g. Toyota, Honda, Suzuki.")
    _models_raw = MAKE_MODELS.get(make, ["Other"])
    _models_sorted = sorted(m for m in _models_raw if m != "Other") + (["Other"] if "Other" in _models_raw else [])
    car_model = st.selectbox("Model", _models_sorted,
                              help="The specific model line — e.g. Corolla, Civic, Alto.")
    min_year  = MODEL_MIN_YEAR.get(car_model, MAKE_MIN_YEAR.get(make, 1990))
    max_year  = datetime.date.today().year
    year      = st.number_input("Model Year", min_value=min_year, max_value=max_year,
                                 value=max(min_year, 2020), step=1,
                                 help=f"The year the car was manufactured. Must be {min_year} or later for this model.")

    st.markdown("**Specs**")
    car_age_ui  = max(max_year - year, 1)
    max_mileage = min(car_age_ui * 30_000, 500_000)
    mileage_km  = st.number_input("Total Mileage (km)", min_value=0, max_value=max_mileage,
                                   value=min(50_000, max_mileage), step=1000,
                                   help=f"Odometer reading — how many kilometres the car has driven in total. Max shown is ~30,000 km/year × car age ({car_age_ui} yr).")
    # Engine range: strictly the stock range this model was ever offered with.
    # "Other" → allow the full known market range since we don't know the car.
    MAKE_ENGINE_RANGE = {
        "Audi": (1400, 5000), "Bmw": (1600, 5000), "Mercedes": (1500, 5500),
        "Mercedes Benz": (1500, 5500), "Lexus": (2500, 5700), "Volkswagen": (1200, 3000),
        "Toyota": (660, 4700), "Honda": (660, 3500), "Suzuki": (660, 1300),
        "Kia": (1000, 3500), "Hyundai": (1000, 3000), "Mitsubishi": (1500, 3500),
        "Nissan": (1000, 3500), "Daihatsu": (660, 1300), "Mazda": (1300, 3000),
        "Subaru": (1500, 3600), "Ford": (1000, 5000), "Jeep": (2000, 4000),
    }
    if car_model == "Other":
        eng_min, eng_max = (660, 6000)
    else:
        make_eng_fallback = MAKE_ENGINE_RANGE.get(make, (660, 6000))
        eng_min, eng_max = MODEL_ENGINE_RANGE.get(car_model, make_eng_fallback)
    eng_default = eng_min  # base / entry variant
    engine_cc = st.number_input(
        "Engine Displacement (cc)",
        min_value=eng_min, max_value=eng_max,
        value=eng_default, step=100,
        help=(
            f"Valid engine range for {car_model}: {eng_min}–{eng_max} cc "
            f"(the smallest to largest engine this model was ever sold with). "
            "Bigger engines mean more power and usually a higher resale price."
        )
    )

with col2:
    st.markdown("**Type & Origin**")
    # When model is "Other", fall back to the make-level lists so irrelevant
    # fuel types (e.g. CNG for Audi, Electric for Suzuki) are never shown.
    # For known models, use the model-level lists first, then fall back to make-level.
    if car_model == "Other":
        valid_fuels = MAKE_FUEL_TYPES.get(make, FUEL_TYPES)
        valid_trans = MAKE_TRANSMISSIONS.get(make, TRANSMISSIONS)
    else:
        valid_fuels = MODEL_FUEL_TYPES.get(car_model, MAKE_FUEL_TYPES.get(make, FUEL_TYPES))
        valid_trans = MODEL_TRANSMISSIONS.get(car_model, MAKE_TRANSMISSIONS.get(make, TRANSMISSIONS))
    fuel_type    = st.selectbox("Fuel Type", valid_fuels,
                                 help="Petrol = most common. Hybrid = petrol + electric motor (better resale). Diesel = heavy vehicles. CNG = compressed gas (cheaper to run, lower resale).")
    transmission = st.selectbox("Transmission", valid_trans,
                                 help="Automatic = self-shifting gears (higher demand, higher price). Manual = driver shifts gears (cheaper, lower resale in Pakistan's city market).")
    assembly     = st.selectbox("Assembly", ASSEMBLIES,
                                 help="Local = assembled in Pakistan (e.g. Indus Motor). Imported = brought from Japan/Europe as a used or new vehicle. Imported cars usually command a premium.")

    st.markdown("**Location**")
    city     = st.selectbox("Listing City", CITIES,
                             help="The city where the car is currently being sold. Metro cities like Karachi/Lahore/Islamabad tend to have higher listing prices.")
    reg_city = st.selectbox("Registration City", CITIES, index=CITIES.index(city),
                             help="The city where the car is officially registered. If different from the listing city, this is a cross-city sale — which can slightly affect price and paperwork cost.")

st.divider()
predict_btn = st.button("🔍 Predict Price", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# PREDICTION + XAI
# ─────────────────────────────────────────────
if predict_btn:
    # ── Validation ──
    valid_models = MAKE_MODELS.get(make, [])
    if car_model not in valid_models:
        st.error(f"'{car_model}' is not a valid model for {make}. Please select a correct combination.")
        st.stop()

    min_year_check = MODEL_MIN_YEAR.get(car_model, MAKE_MIN_YEAR.get(make, 1990))
    if year < min_year_check:
        st.error(f"The {make} {car_model} was not available in {year}. Earliest valid year is {min_year_check}.")
        st.stop()

    # Validate fuel and transmission for both known models and "Other".
    if car_model == "Other":
        valid_fuels_check = MAKE_FUEL_TYPES.get(make, FUEL_TYPES)
        valid_trans_check = MAKE_TRANSMISSIONS.get(make, TRANSMISSIONS)
    else:
        valid_fuels_check = MODEL_FUEL_TYPES.get(car_model, MAKE_FUEL_TYPES.get(make, FUEL_TYPES))
        valid_trans_check = MODEL_TRANSMISSIONS.get(car_model, MAKE_TRANSMISSIONS.get(make, TRANSMISSIONS))

    if fuel_type not in valid_fuels_check:
        st.error(f"The {make} {car_model} does not come in a {fuel_type} variant. Valid options: {', '.join(valid_fuels_check)}.")
        st.stop()

    if transmission not in valid_trans_check:
        st.error(f"The {make} {car_model} is not available with {transmission} transmission. Valid options: {', '.join(valid_trans_check)}.")
        st.stop()

    input_df         = build_input_df(make, car_model, year, mileage_km,
                                      engine_cc, transmission, assembly,
                                      fuel_type, city, reg_city)
    active_model     = info["model"]
    active_explainer = info["explainer"]

    # ── Prediction ──
    price_rs = float(active_model.predict(input_df)[0])
    if price_rs <= 0:
        st.error("The model returned an invalid price for this combination. Please check your inputs.")
        st.stop()

    # ── Price Result Box ──
    low  = max(0, price_rs - MAE)
    high = price_rs + MAE
    st.markdown(f"""
    <div class="price-box">
        <div class="price-label">Estimated Market Price</div>
        <div class="price-main">Rs {price_rs:,.0f}</div>
        <div class="price-range">Expected range: Rs {low:,.0f} – Rs {high:,.0f}</div>
        <div class="price-note">
            {make} {car_model} · {year} · {mileage_km:,} km · {transmission} · {fuel_type} · {assembly}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP ──
    shap_vals = active_explainer.shap_values(input_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_row = shap_vals[0]
    shap_rs  = shap_row  # shape: (22,)

    contributions = sorted(
        zip(READABLE_NAMES, shap_rs),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    top_up   = [(f, v) for f, v in contributions if v >= 0]
    top_down = [(f, v) for f, v in contributions if v < 0]

    # ── Plain-English one-liner per factor (no technical terms) ──
    _age  = max(max_year - year, 0)
    _mpyr = int(mileage_km / max(_age, 1))
    _tier = get_brand_tier(make)
    _ctier = get_city_tier(city)

    def _sentence(feat, v):
        up = v >= 0
        if feat == 'Make':
            return (f"{make} is a well-known, trusted brand — buyers pay more for it."
                    if up else
                    f"{make} is an everyday affordable brand, so buyers expect a lower price compared to luxury names.")
        if feat == 'Model':
            return (f"The {car_model} is popular and in high demand, which keeps its price up."
                    if up else
                    f"The {car_model} is less in demand compared to other models, which brings the price down.")
        if feat == 'Year':
            return (f"It is a {year} — a fairly new car that has not lost much value yet."
                    if up else
                    f"It is a {year} — an older car, and vehicles lose value the longer they have been on the road.")
        if feat == 'Total Mileage (km)':
            return (f"At {mileage_km:,} km the car has not been driven much — lower mileage means less wear, so buyers pay more."
                    if up else
                    f"At {mileage_km:,} km the car has been driven quite a lot. The more a car is driven, the more it wears out, which reduces the price.")
        if feat == 'Engine (cc)':
            return (f"The {engine_cc} cc engine is bigger than most cars in this range, which means more power and a higher price."
                    if up else
                    f"The {engine_cc} cc engine is on the smaller side — good for fuel savings but tends to mean a lower selling price.")
        if feat == 'Assembly':
            return ("Imported cars are generally seen as higher quality by buyers in Pakistan, so they sell for more."
                    if assembly == 'Imported' else
                    "Locally assembled cars are more common, so they typically sell for less than imported ones.")
        if feat == 'Reg. City':
            return (f"Being registered in {reg_city} makes paperwork easy for local buyers, which helps the price."
                    if up else
                    f"The car is registered in {reg_city} but sold elsewhere — buyers have to handle extra paperwork, which can reduce what they are willing to pay.")
        if feat == 'Listing City':
            return (f"Selling in {city} — a big city with lots of buyers — means more demand and a higher price."
                    if up else
                    f"Selling in {city} means a smaller pool of buyers, which tends to bring the price down.")
        if feat == 'Transmission':
            return ("Automatic cars are much more popular in Pakistani cities — easier to drive and buyers willingly pay more for them."
                    if transmission == 'Automatic' else
                    "Manual cars are less popular these days, especially in cities, so they tend to sell for less than automatic ones.")
        if feat == 'Fuel Type':
            if fuel_type in ('Hybrid', 'Phev'):
                return "Hybrid cars save a lot on fuel costs, which makes them very attractive to buyers and pushes the price up."
            elif fuel_type == 'Petrol':
                return ("Petrol is the most common and preferred fuel type — steady demand keeps the price healthy."
                        if up else
                        "Petrol demand is steady but nothing exceptional compared to the most sought-after options.")
            else:
                return f"{fuel_type} fuel has lower buyer demand in Pakistan, which reduces the price."
        if feat == 'Car Age (yrs)':
            yrs = f"{_age} year" + ("s" if _age != 1 else "")
            return (f"At {yrs} old, the car is still relatively young and holds its value well."
                    if up else
                    f"At {yrs} old, the car has aged enough that buyers expect a noticeably lower price.")
        if feat == 'Avg km/year':
            return (f"On average this car was driven {_mpyr:,} km per year — quite low. A lightly used car is in better shape and worth more."
                    if up else
                    f"On average this car was driven {_mpyr:,} km per year — on the higher side. Heavy yearly use means the car wears out faster, reducing its value.")
        if feat == 'Age Group':
            if _age <= 3:
                return "As a nearly new car, it has not depreciated much — buyers still see it as modern."
            elif _age <= 7:
                return ("At this age the car is in the middle of its life — some value has been lost but it is still a solid buy."
                        if up else
                        "At this age the car is in the middle of its life — enough depreciation has happened to noticeably affect the price.")
            else:
                return "This car is getting older, and buyers expect a lower price for vehicles in this age range."
        if feat == 'Imported':
            return ("Imported cars are seen as better quality and fetch a higher price in Pakistan."
                    if assembly == 'Imported' else
                    "Locally assembled cars are priced lower — buyers know they are widely available.")
        if feat == 'Automatic':
            return ("Automatic gears are preferred by most buyers today, especially for city driving — this pushes the price up."
                    if transmission == 'Automatic' else
                    "Manual gears are less popular with buyers now, which brings the resale price down.")
        if feat == 'Cross-City Sale':
            return ("The car is registered and listed in the same city — easy for buyers, no extra hassle."
                    if city == reg_city else
                    "The car is registered in a different city from where it is being sold. Buyers have to handle extra transfer paperwork, which makes them less willing to pay full price.")
        if feat == 'Hybrid':
            return ("Hybrid cars use less fuel, which saves buyers money over time — so people pay more upfront to get one."
                    if fuel_type in ('Hybrid', 'Phev') else
                    f"This car runs entirely on {fuel_type}. Buyers looking for fuel savings may prefer hybrids, which can slightly limit demand.")
        if feat == 'Brand Tier':
            if _tier == 3:
                return ("Audi, BMW, Mercedes and similar brands are luxury names — people pay a big premium just for the brand."
                        if up else
                        "Even luxury brand cars can have factors that pull the price down, such as age or mileage.")
            elif _tier == 2:
                return ("Toyota, Honda, Kia and similar brands are well-regarded and hold their value well."
                        if up else
                        "This brand is respected but not in the luxury tier, so there is a ceiling on how high the price can go.")
            else:
                return "This brand is in the budget segment — great for affordability, but buyers expect a lower price."
        if feat == 'City Tier':
            if _ctier == 1:
                return (f"Big cities like {city} have more buyers and higher living costs, so cars sell for more there."
                        if up else
                        f"Even in a big city like {city}, other factors are pulling this price down.")
            elif _ctier == 2:
                return (f"{city} is a major city with decent demand, though prices are a bit lower than in top metros."
                        if up else
                        f"{city} is a mid-tier city — demand is decent but not as high as in the biggest cities.")
            else:
                return f"{city} is a smaller city — fewer buyers means sellers often have to accept a lower price."
        if feat == 'Engine Category':
            return ("A bigger engine means more power, and powerful cars are priced higher — especially SUVs and luxury vehicles."
                    if up else
                    "A smaller engine keeps running costs low, but buyers looking for power will look elsewhere — which limits the price.")
        if feat == 'Age x Mileage':
            return ("This car is relatively young and has not been driven much — a great combination that keeps the price up."
                    if up else
                    "This car is both old and has been driven a lot. When these two combine, they have a strong effect on lowering the price.")
        if feat == 'Brand x Engine':
            return ("A well-known brand combined with a bigger engine is a very attractive package — buyers pay a significant premium for this."
                    if up else
                    "A budget brand with a smaller engine is at the affordable end of the market — practical, but priced lower.")
        return ""

    # ── Section: What's pushing price UP ──
    if top_up:
        st.markdown('<div class="section-header">🟢 What\'s raising the price</div>', unsafe_allow_html=True)
        for rank, (feat, val) in enumerate(top_up, 1):
            sentence = _sentence(feat, val)
            st.markdown(f"""
            <div class="factor-card up">
                <div>
                    <div class="feat-name">{feat}</div>
                    <div class="feat-desc">{sentence}</div>
                </div>
                <div class="feat-val">+Rs {abs(val):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Section: What's pulling price DOWN ──
    if top_down:
        st.markdown('<div class="section-header">🔴 What\'s lowering the price</div>', unsafe_allow_html=True)
        for rank, (feat, val) in enumerate(top_down, 1):
            sentence = _sentence(feat, val)
            st.markdown(f"""
            <div class="factor-card down">
                <div>
                    <div class="feat-name">{feat}</div>
                    <div class="feat-desc">{sentence}</div>
                </div>
                <div class="feat-val">−Rs {abs(val):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Price Breakdown Chart ──
    st.markdown('<div class="section-header">📊 Price Breakdown Chart</div>', unsafe_allow_html=True)
    st.markdown(
        "Each bar shows how much one factor raised or lowered the price. "
        "**Green = raised the price. Red = lowered the price.** Longer bar = bigger effect."
    )

    top_idx    = np.argsort(np.abs(shap_rs))[::-1][:8]
    plot_names = [READABLE_NAMES[i] for i in top_idx]
    plot_vals  = [shap_rs[i]        for i in top_idx]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = ["#16a34a" if v >= 0 else "#dc2626" for v in plot_vals]
    bars    = ax.barh(plot_names[::-1], plot_vals[::-1],
                      color=colors[::-1], edgecolor="white", height=0.6)

    ax.axvline(0, color="#334155", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Effect on Price (Rs)", fontsize=10, color="#475569")
    ax.set_title(f"What drove the price of this {make} {car_model}?",
                 fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)

    max_abs = max(abs(v) for v in plot_vals) if plot_vals else 1
    for bar, val in zip(bars, plot_vals[::-1]):
        label  = f"{'+'if val>=0 else ''}Rs {val/1_000:.0f}K"
        bar_w  = bar.get_width()
        bar_cy = bar.get_y() + bar.get_height() / 2
        if val >= 0:
            ax.text(bar_w + max_abs * 0.02, bar_cy, label,
                    va="center", ha="left", fontsize=8.5, color="#16a34a", fontweight="bold")
        else:
            ax.text(bar_w - max_abs * 0.02, bar_cy, label,
                    va="center", ha="right", fontsize=8.5, color="#dc2626", fontweight="bold")

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - max_abs * 0.18, x_max + max_abs * 0.18)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs {x/1_000:.0f}K"))
    ax.yaxis.set_tick_params(pad=6)

    pos_patch = mpatches.Patch(color="#16a34a", label="Raises price ↑")
    neg_patch = mpatches.Patch(color="#dc2626", label="Lowers price ↓")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right",
              framealpha=0.8, edgecolor="#e2e8f0")
    fig.subplots_adjust(left=0.22)
    st.pyplot(fig)
    plt.close(fig)
