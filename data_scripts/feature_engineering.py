import pandas as pd
import numpy as np


# 1. LOAD CLEANED DATA


df = pd.read_csv("pakwheels_cleaned.csv")
initial_cols = len(df.columns)
print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")


# 2. TIME-BASED FEATURES


current_year = 2026

df['car_age'] = current_year - df['year']
df['mileage_per_year'] = (df['mileage_km'] / df['car_age'].replace(0, 1)).round(0).astype(int)

def age_group(age):
    if age <= 3:
        return 'New'
    elif age <= 7:
        return 'Mid'
    else:
        return 'Old'

df['age_group'] = df['car_age'].apply(age_group)


# 3. BINARY FEATURES


df['is_imported'] = (df['assembly'] == 'Imported').astype(int)
df['is_automatic'] = (df['transmission'] == 'Automatic').astype(int)
df['cross_city_sale'] = (df['reg_city'] != df['city']).astype(int)
df['is_hybrid'] = (df['fuel_type'] == 'Hybrid').astype(int)


# 4. BRAND TIER


luxury_brands = ['Mercedes', 'Bmw', 'Audi', 'Porsche', 'Lexus', 'Jaguar',
                 'Land Rover', 'Range Rover', 'Bentley', 'Ferrari']

premium_brands = ['Toyota', 'Honda', 'Hyundai', 'Kia', 'Nissan', 'Mazda',
                  'Volkswagen', 'Mitsubishi', 'Changan', 'Mg', 'Haval']

def brand_tier(make):
    make_title = str(make).title()
    if any(lux in make_title for lux in luxury_brands):
        return 3
    elif any(prem in make_title for prem in premium_brands):
        return 2
    else:
        return 1

df['brand_tier'] = df['make'].apply(brand_tier)


# 5. CITY TIER


metro_cities = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi']
major_cities = ['Faisalabad', 'Multan', 'Peshawar', 'Quetta',
                'Sialkot', 'Gujranwala', 'Hyderabad']

def city_tier(city):
    city_title = str(city).title()
    if city_title in metro_cities:
        return 1
    elif city_title in major_cities:
        return 2
    else:
        return 3

df['city_tier'] = df['city'].apply(city_tier)


# 6. PRICE-BASED FEATURES


df['price_per_cc'] = (df['price_pkr'] / df['engine_cc']).round(0).astype(int)
df['log_price'] = np.log1p(df['price_pkr'])

def price_category(price):
    if price < 2_000_000:
        return 'Budget'
    elif price < 5_000_000:
        return 'Mid-Range'
    elif price < 10_000_000:
        return 'Premium'
    else:
        return 'Luxury'

df['price_category'] = df['price_pkr'].apply(price_category)


# 7. ENGINE CATEGORY


def engine_category(cc):
    if cc < 1000:
        return 'Small'
    elif cc < 1600:
        return 'Medium'
    elif cc < 2500:
        return 'Large'
    else:
        return 'Very Large'

df['engine_category'] = df['engine_cc'].apply(engine_category)


# 8. INTERACTION FEATURES


df['age_mileage_interact'] = df['car_age'] * df['mileage_km']
df['brand_engine_interact'] = df['brand_tier'] * df['engine_cc']


# 9. SAVE & SUMMARY


df.to_csv("pakwheels_features.csv", index=False)

print(f"New features added: {len(df.columns) - initial_cols}")
print(f"Total columns: {len(df.columns)}")
print(f"Saved to: pakwheels_features.csv")