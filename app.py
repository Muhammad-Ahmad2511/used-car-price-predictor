import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    xgb_model      = joblib.load("models/xgboost_model.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    lgbm_explainer = shap.TreeExplainer(lgbm_model)
    xgb_explainer  = shap.TreeExplainer(xgb_model)
    return lgbm_model, xgb_model, label_encoders, lgbm_explainer, xgb_explainer

lgbm_model, xgb_model, label_encoders, lgbm_explainer, xgb_explainer = load_artifacts()

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
    "LightGBM": {"r2": 0.873, "mae": "Rs 562K", "model": lgbm_model, "explainer": lgbm_explainer},
    "XGBoost":  {"r2": 0.861, "mae": "Rs 571K", "model": xgb_model,  "explainer": xgb_explainer},
}

# ─────────────────────────────────────────────
# OPTION LISTS
# ─────────────────────────────────────────────
MAKES = [
    "Toyota", "Honda", "Suzuki", "Kia", "Hyundai", "Daihatsu",
    "Nissan", "Mitsubishi", "Mg", "Haval", "Changan", "Chery",
    "Audi", "Bmw", "Mercedes", "Mercedes Benz", "Lexus",
    "Volkswagen", "Mazda", "Subaru", "Ford", "Jeep",
    "Proton", "Faw", "Dfsk", "Jac", "Byd", "Tesla", "Other"
]

TOP_MODELS = [
    "Corolla", "Corolla Gli 1.3 Vvti", "Corolla Gli Automatic 1.6 Vvti",
    "Corolla Altis 1.8", "Corolla Altis Grande 1.8", "Corolla Xli Vvti",
    "Civic", "Civic Oriel", "Civic Vti Oriel 1.8 I-Vtec", "Civic Rs",
    "Civic 1.5 Rs Turbo", "City", "City 1.3 I-Vtec Prosmatec",
    "City 1.5L Cvt", "City 1.5L Aspire Cvt",
    "Alto", "Alto Vxr", "Alto Vxl Ags", "Alto Vxr (Cng)",
    "Wagon R", "Wagon R Vxr", "Wagon R Vxl", "Wagon R Ags",
    "Cultus", "Cultus Vxl", "Cultus Vxr", "Cultus Vxli",
    "Swift", "Swift Dlx 1.3", "Swift Gl Cvt",
    "Prius", "Prius S 1.8", "Prius A", "Aqua", "Aqua G",
    "Vitz", "Vitz F 1.0", "Vitz F 1.3",
    "Fortuner", "Fortuner 2.7 V", "Fortuner Legender",
    "Prado", "Prado Tx 2.7", "Prado Tz G 4.0",
    "Land Cruiser", "Land Cruiser Zx", "Land Cruiser Vx 4.7",
    "Sportage", "Sportage Alpha", "Sportage Fwd", "Sportage Awd",
    "Tucson", "Tucson Fwd A/T Gls", "Tucson Hybrid Smart",
    "Mira", "Mira G Sa Iii", "Mira X", "Cuore", "Cuore Cx",
    "Hilux", "Hilux Revo V Automatic 2.8", "Hilux Revo G Automatic 2.8",
    "Mehran Vx", "Mehran Vxr", "Bolan Vx",
    "Vezel", "Vezel Hybrid X", "Vezel E-Hev Z",
    "Hr-V", "Cr-V", "Br-V", "Br-V I-Vtec",
    "Picanto", "Carnival", "Pajero", "Pajero Exceed 3.5",
    "Lancer Glx 1.5", "Lancer Glx 1.6", "Other"
]

CITIES = [
    "Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad",
    "Multan", "Peshawar", "Quetta", "Sialkot", "Gujranwala",
    "Hyderabad", "Gujrat", "Sargodha", "Bahawalpur", "Other"
]

TRANSMISSIONS = ["Automatic", "Manual"]
ASSEMBLIES    = ["Local", "Imported"]
FUEL_TYPES    = ["Petrol", "Hybrid", "Diesel", "CNG", "Electric", "Lpg", "Phev"]

# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors feature_engineering.py)
# ─────────────────────────────────────────────
LUXURY_BRANDS  = ['Mercedes', 'Mercedes Benz', 'Bmw', 'Audi', 'Porsche', 'Lexus',
                  'Jaguar', 'Land', 'Land Rover', 'Range', 'Range Rover', 'Bentley']
PREMIUM_BRANDS = ['Toyota', 'Honda', 'Hyundai', 'Kia', 'Nissan', 'Mazda',
                  'Volkswagen', 'Mitsubishi', 'Changan', 'Mg', 'Haval']
METRO_CITIES   = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi']
MAJOR_CITIES   = ['Faisalabad', 'Multan', 'Peshawar', 'Quetta',
                  'Sialkot', 'Gujranwala', 'Hyderabad']

def get_brand_tier(make: str) -> int:
    m = str(make).title()
    if any(lb in m for lb in LUXURY_BRANDS):  return 3
    if any(pb in m for pb in PREMIUM_BRANDS): return 2
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
def safe_encode(le, val: str) -> int:
    return int(le.transform([val])[0]) if val in le.classes_ else 0

def build_input_df(make, car_model, year, mileage_km, engine_cc,
                   transmission, assembly, fuel_type, city, reg_city) -> pd.DataFrame:
    current_year     = 2026
    car_age          = current_year - year
    mileage_per_year = int(mileage_km / max(car_age, 1))
    age_mileage_int  = car_age * mileage_km
    bt               = get_brand_tier(make)
    brand_engine_int = bt * engine_cc
    is_imported      = 1 if assembly == "Imported"      else 0
    is_automatic     = 1 if transmission == "Automatic" else 0
    is_hybrid        = 1 if fuel_type == "Hybrid"       else 0
    cross_city_sale  = 1 if city != reg_city            else 0
    city_tier        = get_city_tier(city)

    le = label_encoders
    row = {
        'make':                  safe_encode(le['make'],            make),
        'model':                 safe_encode(le['model'],           car_model),
        'year':                  year,
        'mileage_km':            mileage_km,
        'engine_cc':             engine_cc,
        'assembly':              safe_encode(le['assembly'],        assembly),
        'reg_city':              safe_encode(le['reg_city'],        reg_city),
        'city':                  safe_encode(le['city'],            city),
        'transmission':          safe_encode(le['transmission'],    transmission),
        'fuel_type':             safe_encode(le['fuel_type'],       fuel_type),
        'car_age':               car_age,
        'mileage_per_year':      mileage_per_year,
        'age_group':             safe_encode(le['age_group'],       get_age_group(car_age)),
        'is_imported':           is_imported,
        'is_automatic':          is_automatic,
        'cross_city_sale':       cross_city_sale,
        'is_hybrid':             is_hybrid,
        'brand_tier':            bt,
        'city_tier':             city_tier,
        'engine_category':       safe_encode(le['engine_category'], get_engine_category(engine_cc)),
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
# UI — HEADER
# ─────────────────────────────────────────────
st.title("Car Price Predictor - PakWheels")
st.markdown(
    "Enter car details and click **Predict Price** for an AI-powered "
    "estimate with a full SHAP explanation."
)
st.divider()

# ─────────────────────────────────────────────
# MODEL SELECTOR
# ─────────────────────────────────────────────
st.subheader("Model Selection")
model_choice = st.radio(
    "Choose prediction model:",
    ["LightGBM", "XGBoost"],
    horizontal=True,
)
info = MODEL_INFO[model_choice]
st.info(
    f"**{model_choice}** — Test R2 = **{info['r2']}**  |  "
    f"Test MAE = **{info['mae']}**"
)
st.divider()

# ─────────────────────────────────────────────
# UI — INPUT FORM
# ─────────────────────────────────────────────
st.subheader("Car Details")
col1, col2 = st.columns(2)

with col1:
    make       = st.selectbox("Make",            sorted(MAKES))
    car_model  = st.selectbox("Model",           sorted(TOP_MODELS))
    year       = st.number_input("Year",          min_value=1990, max_value=2026, value=2020, step=1)
    mileage_km = st.number_input("Mileage (km)",  min_value=0,    max_value=500_000, value=50_000, step=1000)
    engine_cc  = st.number_input("Engine (cc)",   min_value=100,  max_value=10_000,  value=1300,   step=100)

with col2:
    fuel_type    = st.selectbox("Fuel Type",         FUEL_TYPES)
    transmission = st.selectbox("Transmission",      TRANSMISSIONS)
    assembly     = st.selectbox("Assembly",          ASSEMBLIES)
    city         = st.selectbox("Listing City",      CITIES)
    reg_city     = st.selectbox("Registration City", CITIES, index=0)

st.divider()
predict_btn = st.button("Predict Price", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# PREDICTION + XAI
# ─────────────────────────────────────────────
if predict_btn:
    input_df         = build_input_df(make, car_model, year, mileage_km,
                                      engine_cc, transmission, assembly,
                                      fuel_type, city, reg_city)
    active_model     = info["model"]
    active_explainer = info["explainer"]

    # ── Prediction ──
    # Both models predict raw price_pkr directly — no log/expm1 transform
    price_rs = float(active_model.predict(input_df)[0])

    st.success(f"### Estimated Price: Rs {price_rs:,.0f}")
    st.caption(
        f"**{model_choice}** · "
        f"Approx. range: Rs {price_rs * 0.88:,.0f} - Rs {price_rs * 1.12:,.0f}  "
        f"(Test R2 = {info['r2']})"
    )

    # ── SHAP ──
    shap_vals = active_explainer.shap_values(input_df)
    shap_row  = shap_vals[0]
    # Both models predict raw PKR, so SHAP values are already in Rs
    base_rs   = float(active_explainer.expected_value)
    shap_rs   = shap_row  # shape: (22,)

    # ── Plain-text summary ──
    st.subheader("Why this price?")
    contributions = sorted(
        zip(READABLE_NAMES, shap_rs),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    parts = []
    for feat, val in contributions:
        sign = "+" if val >= 0 else "-"
        parts.append(f"**{feat}** ({sign}Rs {abs(val):,.0f})")

    st.markdown(
        f"This car is estimated at **Rs {price_rs:,.0f}** by {model_choice}. "
        f"The 5 biggest price drivers are: {', '.join(parts)}."
    )

    # ── Top-5 factor cards ──
    st.subheader("Top 5 Price Factors")
    for feat, val in contributions:
        color_hex = "#1a9c3e" if val >= 0 else "#d62728"
        arrow     = "up" if val >= 0 else "down"
        sign      = "+" if val >= 0 else "-"
        st.markdown(
            f"""
            <div style="
                border-left: 5px solid {color_hex};
                background: #f9f9f9;
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            ">
                <span style="font-weight:600">{feat}</span>
                &nbsp;&nbsp;
                <span style="color:{color_hex}; font-weight:700">
                    {sign}Rs {abs(val):,.0f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Waterfall bar chart ──
    st.subheader("Price Breakdown Chart")
    top_idx    = np.argsort(np.abs(shap_rs))[::-1][:8]
    plot_names = [READABLE_NAMES[i] for i in top_idx]
    plot_vals  = [shap_rs[i]        for i in top_idx]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = ["#1a9c3e" if v >= 0 else "#d62728" for v in plot_vals]
    bars    = ax.barh(plot_names[::-1], plot_vals[::-1],
                      color=colors[::-1], edgecolor="white", height=0.6)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to Price (Rs)", fontsize=10)
    ax.set_title(f"SHAP Feature Contributions - {model_choice}",
                 fontsize=12, fontweight="bold")

    max_abs = max(abs(v) for v in plot_vals) if plot_vals else 1
    for bar, val in zip(bars, plot_vals[::-1]):
        label  = f"{'+'if val>=0 else ''}Rs {val/1_000:.0f}K"
        bar_w  = bar.get_width()
        bar_cy = bar.get_y() + bar.get_height() / 2
        # Place label OUTSIDE the bar end, away from the Y-axis
        if val >= 0:
            # positive bars: label to the right of the bar end
            ax.text(bar_w + max_abs * 0.02, bar_cy, label,
                    va="center", ha="left", fontsize=8.5, color="#1a9c3e", fontweight="bold")
        else:
            # negative bars: label to the LEFT of the bar start (which is bar_w, a negative number)
            # but keep it well clear of the Y-axis by anchoring from the right edge of the bar
            ax.text(bar_w - max_abs * 0.02, bar_cy, label,
                    va="center", ha="right", fontsize=8.5, color="#d62728", fontweight="bold")

    # Expand x-axis limits so labels don't get clipped
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - max_abs * 0.18, x_max + max_abs * 0.18)

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rs {x / 1_000:.0f}K")
    )
    # Give Y-axis labels more breathing room
    ax.yaxis.set_tick_params(pad=6)

    pos_patch = mpatches.Patch(color="#1a9c3e", label="Increases price")
    neg_patch = mpatches.Patch(color="#d62728", label="Decreases price")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")
    fig.subplots_adjust(left=0.22)   # extra left margin for feature names
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Baseline (average car in dataset): **Rs {base_rs:,.0f}**. "
        f"SHAP values show how each feature shifts the price from this baseline."
    )
