import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

Path("eda/plots").mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# LOAD DATA


df = pd.read_csv("pakwheels_features.csv")
print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

# Normalize transmission for clean plotting
def normalize_transmission(t):
    t = str(t).lower()
    if 'auto' in t:
        return 'Automatic'
    elif 'manual' in t:
        return 'Manual'
    else:
        return 'Other'

df['transmission_plot'] = df['transmission'].apply(normalize_transmission)


# PLOT 1: PRICE DISTRIBUTION


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(df['price_pkr'] / 1_000_000, bins=50, edgecolor='black', color='steelblue')
ax1.set_xlabel('Price (Million PKR)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Price Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2.boxplot(df['price_pkr'] / 1_000_000, vert=True,
            flierprops=dict(marker='o', markersize=2, alpha=0.3))
ax2.set_ylabel('Price (Million PKR)', fontsize=12)
ax2.set_title('Price Boxplot', fontsize=14, fontweight='bold')
ax2.set_xticks([])
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda/plots/price_distribution.png', bbox_inches='tight')
plt.close()


# PLOT 2: FEATURE DISTRIBUTIONS


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

axes[0].hist(df['year'], bins=30, edgecolor='black', color='coral')
axes[0].set_xlabel('Year', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Year Distribution', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

axes[1].hist(df['mileage_km'] / 1000, bins=40, edgecolor='black', color='lightgreen')
axes[1].set_xlabel('Mileage (1000 km)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Mileage Distribution', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

axes[2].hist(df['engine_cc'], bins=30, edgecolor='black', color='gold')
axes[2].set_xlabel('Engine (cc)', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title('Engine Distribution', fontsize=12, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

axes[3].hist(df['car_age'], bins=30, edgecolor='black', color='plum')
axes[3].set_xlabel('Car Age (years)', fontsize=11)
axes[3].set_ylabel('Frequency', fontsize=11)
axes[3].set_title('Car Age Distribution', fontsize=12, fontweight='bold')
axes[3].grid(axis='y', alpha=0.3)

axes[4].hist(df['mileage_per_year'], bins=40, edgecolor='black', color='skyblue')
axes[4].set_xlabel('Mileage per Year (km)', fontsize=11)
axes[4].set_ylabel('Frequency', fontsize=11)
axes[4].set_title('Mileage/Year Distribution', fontsize=12, fontweight='bold')
axes[4].grid(axis='y', alpha=0.3)

tier_counts = df['brand_tier'].value_counts().sort_index()
axes[5].bar([1, 2, 3], [tier_counts.get(i, 0) for i in [1, 2, 3]],
            color=['lightcoral', 'lightblue', 'gold'], edgecolor='black')
axes[5].set_xlabel('Brand Tier', fontsize=11)
axes[5].set_ylabel('Count', fontsize=11)
axes[5].set_title('Brand Tier Distribution', fontsize=12, fontweight='bold')
axes[5].set_xticks([1, 2, 3])
axes[5].set_xticklabels(['Budget', 'Premium', 'Luxury'])
axes[5].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda/plots/feature_distributions.png', bbox_inches='tight')
plt.close()


# PLOT 3: CATEGORICAL DISTRIBUTIONS


fig, axes = plt.subplots(2, 2, figsize=(13, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

top_makes = df['make'].value_counts().head(10)
axes[0, 0].barh(top_makes.index[::-1], top_makes.values[::-1],
                color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Count', fontsize=11)
axes[0, 0].set_title('Top 10 Car Makes', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)
for i, v in enumerate(top_makes.values[::-1]):
    axes[0, 0].text(v + 30, i, str(v), va='center', fontsize=9)

assembly_counts = df['assembly'].value_counts()
axes[0, 1].pie(assembly_counts.values, labels=assembly_counts.index,
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'],
               startangle=90, textprops={'fontsize': 11})
axes[0, 1].set_title('Assembly Distribution', fontsize=12, fontweight='bold')

trans_counts = df['transmission_plot'].value_counts()
trans_colors = ['skyblue', 'gold', 'lightgray'][:len(trans_counts)]
axes[1, 0].pie(trans_counts.values, labels=trans_counts.index,
               autopct='%1.1f%%', colors=trans_colors,
               startangle=90, textprops={'fontsize': 11})
axes[1, 0].set_title('Transmission Distribution', fontsize=12, fontweight='bold')

fuel_counts = df['fuel_type'].value_counts().head(6)
axes[1, 1].bar(range(len(fuel_counts)), fuel_counts.values,
               color='coral', edgecolor='black')
axes[1, 1].set_xticks(range(len(fuel_counts)))
axes[1, 1].set_xticklabels(fuel_counts.index, rotation=25, ha='right', fontsize=10)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Fuel Type Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(fuel_counts.values):
    axes[1, 1].text(i, v + 50, str(v), ha='center', fontsize=9)

plt.savefig('eda/plots/categorical_distributions.png', bbox_inches='tight')
plt.close()


# PLOT 4: CORRELATION MATRIX


numeric_cols = ['year', 'mileage_km', 'engine_cc', 'car_age', 'mileage_per_year', 'price_pkr']
corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numeric Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('eda/plots/correlation_matrix.png', bbox_inches='tight')
plt.close()


# PLOT 5: PRICE VS FEATURES


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()
plt.subplots_adjust(hspace=0.35, wspace=0.3, bottom=0.08)

axes[0].scatter(df['year'], df['price_pkr'] / 1_000_000,
                alpha=0.3, s=8, color='steelblue')
axes[0].set_xlabel('Year', fontsize=11)
axes[0].set_ylabel('Price (Million PKR)', fontsize=11)
axes[0].set_title('Price vs Year', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].scatter(df['mileage_km'] / 1000, df['price_pkr'] / 1_000_000,
                alpha=0.3, s=8, color='coral')
axes[1].set_xlabel('Mileage (1000 km)', fontsize=11)
axes[1].set_ylabel('Price (Million PKR)', fontsize=11)
axes[1].set_title('Price vs Mileage', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

axes[2].scatter(df['engine_cc'], df['price_pkr'] / 1_000_000,
                alpha=0.3, s=8, color='green')
axes[2].set_xlabel('Engine (cc)', fontsize=11)
axes[2].set_ylabel('Price (Million PKR)', fontsize=11)
axes[2].set_title('Price vs Engine', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

assembly_price = df.groupby('assembly')['price_pkr'].mean() / 1_000_000
axes[3].bar(assembly_price.index, assembly_price.values,
            color=['lightgreen', 'lightcoral'], edgecolor='black', width=0.5)
axes[3].set_xlabel('Assembly', fontsize=11)
axes[3].set_ylabel('Avg Price (Million PKR)', fontsize=11)
axes[3].set_title('Average Price by Assembly', fontsize=12, fontweight='bold')
axes[3].grid(axis='y', alpha=0.3)
for i, v in enumerate(assembly_price.values):
    axes[3].text(i, v + 0.1, f'{v:.1f}M', ha='center', fontsize=10)

trans_price = df.groupby('transmission_plot')['price_pkr'].mean() / 1_000_000
trans_price = trans_price[trans_price.index.isin(['Automatic', 'Manual'])]
axes[4].bar(trans_price.index, trans_price.values,
            color=['skyblue', 'gold'], edgecolor='black', width=0.5)
axes[4].set_xlabel('Transmission', fontsize=11)
axes[4].set_ylabel('Avg Price (Million PKR)', fontsize=11)
axes[4].set_title('Average Price by Transmission', fontsize=12, fontweight='bold')
axes[4].grid(axis='y', alpha=0.3)
for i, v in enumerate(trans_price.values):
    axes[4].text(i, v + 0.1, f'{v:.1f}M', ha='center', fontsize=10)

tier_price = df.groupby('brand_tier')['price_pkr'].mean() / 1_000_000
axes[5].bar([1, 2, 3], [tier_price.get(i, 0) for i in [1, 2, 3]],
            color=['lightcoral', 'lightblue', 'gold'], edgecolor='black', width=0.5)
axes[5].set_xlabel('Brand Tier', fontsize=11)
axes[5].set_ylabel('Avg Price (Million PKR)', fontsize=11)
axes[5].set_title('Average Price by Brand Tier', fontsize=12, fontweight='bold')
axes[5].set_xticks([1, 2, 3])
axes[5].set_xticklabels(['Budget', 'Premium', 'Luxury'])
axes[5].grid(axis='y', alpha=0.3)
for i, t in enumerate([1, 2, 3]):
    v = tier_price.get(t, 0)
    axes[5].text(t, v + 0.1, f'{v:.1f}M', ha='center', fontsize=10)

plt.savefig('eda/plots/price_vs_features.png', bbox_inches='tight')
plt.close()


# SUMMARY STATISTICS


print(f"\nPrice (PKR):  min={df['price_pkr'].min():,}  max={df['price_pkr'].max():,}  mean={df['price_pkr'].mean():,.0f}  median={df['price_pkr'].median():,.0f}")
print(f"Car Age:      min={df['car_age'].min()}  max={df['car_age'].max()}  mean={df['car_age'].mean():.1f}")
print(f"Mileage (km): min={df['mileage_km'].min():,}  max={df['mileage_km'].max():,}  mean={df['mileage_km'].mean():,.0f}")

print(f"\nTop 5 Brands:")
for make, count in df['make'].value_counts().head().items():
    print(f"  {make:<15} {count:>5} ({count/len(df)*100:.1f}%)")

for label, col in [("Assembly", 'assembly'), ("Transmission", 'transmission_plot'), ("Fuel Type", 'fuel_type')]:
    print(f"\n{label}:")
    for val, count in df[col].value_counts().items():
        print(f"  {val:<15} {count:>5} ({count/len(df)*100:.1f}%)")

print("\n5 plots saved to: eda/plots/")