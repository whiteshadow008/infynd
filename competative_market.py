# ===============================================================
# Interactive Fintech Market Predictor (Flask UI) - FIXED
# ===============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
from flask import Flask, render_template, request
warnings.filterwarnings('ignore')

# ---------------------------
# 1️⃣ Original code: Load CSV & feature engineering
# ---------------------------
print("="*70)
print("🚀 FINTECH MARKET PREDICTOR - INITIALIZING")
print("="*70)

df = pd.read_csv("synthetic_fintech_200rows.csv")

def find_column(df, possible_names):
    for col in df.columns:
        for name in possible_names:
            if name.lower() in col.lower():
                return col
    return None

col_mapping = {
    'company': find_column(df, ['company', 'name', 'firm']),
    'category': find_column(df, ['category', 'type', 'segment']),
    'market_share': find_column(df, ['market share', 'share', 'market_share']),
    'transaction_fee': find_column(df, ['transaction fee', 'fee', 'avg fee', 'average fee']),
    'customer_base': find_column(df, ['customer base', 'customers', 'customer_base', 'users']),
    'revenue': find_column(df, ['revenue', 'sales', 'turnover']),
    'pricing_index': find_column(df, ['pricing index', 'price index', 'pricing_index']),
    'growth_rate': find_column(df, ['growth rate', 'growth', 'growth_rate']),
    'pricing_tier': find_column(df, ['pricing tier', 'tier', 'pricing_tier', 'plan']),
    'inflation_adj': find_column(df, ['inflation adj', 'adjusted', 'inflation_adjusted'])
}

# Standardize column names
df_clean = df.copy()
df_clean.rename(columns={
    col_mapping['company']: 'Company',
    col_mapping['category']: 'Category',
    col_mapping['market_share']: 'Market_Share',
    col_mapping['transaction_fee']: 'Transaction_Fee',
    col_mapping['customer_base']: 'Customer_Base',
    col_mapping['revenue']: 'Revenue',
    col_mapping['pricing_index']: 'Pricing_Index',
    col_mapping['growth_rate']: 'Growth_Rate',
    col_mapping['pricing_tier']: 'Pricing_Tier'
}, inplace=True)

if col_mapping['inflation_adj']:
    df_clean.rename(columns={col_mapping['inflation_adj']: 'Inflation_Adj'}, inplace=True)
    df_clean["Inflation_Impact_Pct"] = ((df_clean["Inflation_Adj"] - df_clean["Transaction_Fee"]) /
                                         df_clean["Transaction_Fee"]) * 100
else:
    df_clean["Inflation_Impact_Pct"] = 0

df_clean["Revenue_Per_Customer"] = df_clean["Revenue"] / df_clean["Customer_Base"]
df_clean["Market_Efficiency"] = df_clean["Market_Share"] / df_clean["Transaction_Fee"]
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

# Encode categorical variables
le_category = LabelEncoder()
le_tier = LabelEncoder()
df_clean["Category_Encoded"] = le_category.fit_transform(df_clean["Category"].astype(str))
if 'Pricing_Tier' in df_clean.columns:
    df_clean["Pricing_Tier_Encoded"] = le_tier.fit_transform(df_clean["Pricing_Tier"].astype(str))
    pricing_tiers = df_clean["Pricing_Tier"].unique().tolist()
else:
    df_clean["Pricing_Tier_Encoded"] = 0
    pricing_tiers = ["Standard"]
categories = df_clean["Category"].unique().tolist()

# ---------------------------
# 2️⃣ Train Models
# ---------------------------
features_market = ["Transaction_Fee", "Customer_Base", "Revenue", "Growth_Rate", "Category_Encoded"]
if 'Pricing_Index' in df_clean.columns:
    features_market.append("Pricing_Index")
if 'Pricing_Tier_Encoded' in df_clean.columns:
    features_market.append("Pricing_Tier_Encoded")
X_market = df_clean[features_market]
y_market = df_clean["Market_Share"]
market_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
market_model.fit(X_market, y_market)

features_revenue = ["Market_Share", "Customer_Base", "Transaction_Fee", "Growth_Rate", "Category_Encoded"]
if 'Pricing_Tier_Encoded' in df_clean.columns:
    features_revenue.append("Pricing_Tier_Encoded")
X_revenue = df_clean[features_revenue]
y_revenue = df_clean["Revenue"]
revenue_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
revenue_model.fit(X_revenue, y_revenue)

# ---------------------------
# 3️⃣ Flask Setup
# ---------------------------
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get form data from index2.html - FIX: use lowercase keys
        user_data = {
            'Category': request.form['category'],  # FIXED: was 'Category'
            'Pricing_Tier': request.form['pricing_tier'],  # FIXED: was 'Pricing_Tier'
            'Transaction_Fee': float(request.form['transaction_fee']),
            'Customer_Base': float(request.form['customer_base']),
            'Revenue': float(request.form['revenue']),
            'Pricing_Index': float(request.form['pricing_index']),
            'Growth_Rate': float(request.form['growth_rate'])
        }

        # Use your existing predict function
        def predict_company_performance_for_flask(user_data):
            market_input = [
                user_data['Transaction_Fee'],
                user_data['Customer_Base'],
                user_data['Revenue'],
                user_data['Growth_Rate'],
                le_category.transform([user_data['Category']])[0]
            ]
            if 'Pricing_Index' in df_clean.columns:
                market_input.append(user_data['Pricing_Index'])
            if 'Pricing_Tier_Encoded' in df_clean.columns:
                market_input.append(le_tier.transform([user_data['Pricing_Tier']])[0])
            predicted_market_share = market_model.predict([market_input])[0]

            revenue_input = [
                predicted_market_share,
                user_data['Customer_Base'],
                user_data['Transaction_Fee'],
                user_data['Growth_Rate'],
                le_category.transform([user_data['Category']])[0]
            ]
            if 'Pricing_Tier_Encoded' in df_clean.columns:
                revenue_input.append(le_tier.transform([user_data['Pricing_Tier']])[0])
            predicted_revenue = revenue_model.predict([revenue_input])[0]

            return round(predicted_market_share,2), round(predicted_revenue,2)

        try:
            market_share, revenue = predict_company_performance_for_flask(user_data)
            prediction = {"Market_Share": market_share, "Revenue": revenue}
            print(f"✓ Prediction successful! Market Share: {market_share}%, Revenue: ₹{revenue}")
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            prediction = {"error": str(e)}

    return render_template("index_2.html", categories=categories, pricing_tiers=pricing_tiers, prediction=prediction)

# ---------------------------
# 4️⃣ Run Flask App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port = 4000)