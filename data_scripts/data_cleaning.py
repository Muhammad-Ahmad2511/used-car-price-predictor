import pandas as pd
import numpy as np
import re


# 1. LOAD DATA


df = pd.read_csv("pakwheels_scraped.csv")
initial_rows = len(df)
print(f"Loaded: {len(df):,} rows")


# 2. REMOVE DUPLICATES


before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df):,}")


# 3. HANDLE MISSING VALUES

# Drop rows with missing critical columns
before = len(df)
df.dropna(subset=['make', 'model', 'price_pkr'], inplace=True)
print(f"Rows dropped (missing make/model/price): {before - len(df)}")

# Fill year with median by make
if df['year'].isnull().sum() > 0:
    df['year'] = df.groupby('make')['year'].transform(lambda x: x.fillna(x.median()))
    df['year'] = df['year'].fillna(df['year'].median())

# Fill mileage with median by year
if df['mileage_km'].isnull().sum() > 0:
    df['mileage_km'] = df.groupby('year')['mileage_km'].transform(lambda x: x.fillna(x.median()))
    df['mileage_km'] = df['mileage_km'].fillna(df['mileage_km'].median())

# Fill engine_cc with median by make+model, then make, then default 1300
if df['engine_cc'].isnull().sum() > 0:
    df['engine_cc'] = df.groupby(['make', 'model'])['engine_cc'].transform(lambda x: x.fillna(x.median()))
    df['engine_cc'] = df.groupby('make')['engine_cc'].transform(lambda x: x.fillna(x.median()))
    df['engine_cc'] = df['engine_cc'].fillna(1300)

# Fill assembly with mode by make
if df['assembly'].isnull().sum() > 0:
    df['assembly'] = df.groupby('make')['assembly'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Local')
    )
    df['assembly'] = df['assembly'].fillna('Local')

# Fill remaining categoricals
if df['transmission'].isnull().sum() > 0:
    df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])

if df['fuel_type'].isnull().sum() > 0:
    df['fuel_type'] = df['fuel_type'].fillna('Petrol')

df['city'] = df['city'].fillna('Other')
df['reg_city'] = df['reg_city'].fillna(df['city'])
df['city'] = df['city'].fillna(df['reg_city'])

print(f"Missing values remaining: {df.isnull().sum().sum()}")


# 4. FIX DATA TYPES


df['year'] = df['year'].astype(int)
df['mileage_km'] = df['mileage_km'].astype(int)
df['engine_cc'] = df['engine_cc'].astype(int)
df['price_pkr'] = df['price_pkr'].astype(int)

df['make'] = df['make'].str.strip().str.title()
df['model'] = df['model'].str.strip().str.title()
df['city'] = df['city'].str.strip().str.title()
df['reg_city'] = df['reg_city'].str.strip().str.title()
df['assembly'] = df['assembly'].str.strip().str.title()
df['transmission'] = df['transmission'].str.strip().str.title()
df['fuel_type'] = df['fuel_type'].str.strip().str.title()


# 5. REMOVE OUTLIERS


before = len(df)
df = df[(df['price_pkr'] >= 100_000) & (df['price_pkr'] <= 50_000_000)]
df = df[(df['year'] >= 1990) & (df['year'] <= 2026)]
df = df[(df['mileage_km'] >= 0) & (df['mileage_km'] <= 500_000)]
df = df[(df['engine_cc'] >= 100) & (df['engine_cc'] <= 10_000)]
print(f"Outliers removed: {before - len(df):,}")


# 6. CLEAN MODEL NAMES


def clean_model_name(model):
    model = str(model)
    for pattern in [r'\s+for\s+sale.*', r'\s+in\s+[A-Z][a-z]+.*', r'\s+urgent.*', r'\s+hot\s+deal.*']:
        model = re.sub(pattern, '', model, flags=re.IGNORECASE)
    return model.strip()

df['model'] = df['model'].apply(clean_model_name)


# 7. STANDARDIZE CATEGORIES


df['assembly'] = df['assembly'].replace({'Import': 'Imported', 'Locally': 'Local'})
df['fuel_type'] = df['fuel_type'].replace({'Gasoline': 'Petrol', 'Cng': 'CNG'})
df['transmission'] = df['transmission'].replace({'Auto': 'Automatic'})


# 8. SAVE & SUMMARY


df.to_csv("pakwheels_cleaned.csv", index=False)

print(f"\nCleaning Summary")
print(f"  Initial rows:   {initial_rows:,}")
print(f"  Final rows:     {len(df):,}")
print(f"  Rows removed:   {initial_rows - len(df):,}")
print(f"  Data retained:  {(len(df) / initial_rows) * 100:.1f}%")
print(f"\nPrice range: Rs {df['price_pkr'].min():,} - Rs {df['price_pkr'].max():,}")
print(f"Saved to: pakwheels_cleaned.csv")