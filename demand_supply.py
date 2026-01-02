from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

# ------------------------------
# Load dataset and train model
# ------------------------------
df = pd.read_csv("B2B_Supply_Demand_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['DayOfYear'] = df['Date'].dt.dayofyear

# Create aggregated training data by Product and Region
df_agg = df.groupby(['Product', 'Region', 'Date']).agg({
    'Quantity_Ordered': 'sum',
    'Inventory_Level': 'mean',
    'Year': 'first',
    'Month': 'first',
    'Week': 'first',
    'DayOfYear': 'first'
}).reset_index()

# Prepare features for model training
X = df_agg[['Year', 'Month', 'Week', 'DayOfYear']]
y = df_agg['Quantity_Ordered']

# Train model
model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
model.fit(X, y)

# ------------------------------
# Flask routes
# ------------------------------
@app.route('/', methods=['GET','POST'])
def index():
    summary_df = None
    plot_url = None
    future_date = None
    error_message = None
    top_demand_product = None
    shortage_count = 0
    overstock_count = 0
    balanced_count = 0

    if request.method == 'POST':
        # Safe retrieval of future_date
        future_date_str = request.form.get('future_date', None)
        
        if not future_date_str:
            error_message = "Please select a future date."
            return render_template('index.html', 
                                 summary_table=summary_df, 
                                 plot_url=plot_url, 
                                 future_date=future_date,
                                 error_message=error_message)
        
        try:
            future_date = pd.to_datetime(future_date_str)
            
            # Get unique products and regions
            unique_products = df['Product'].unique()
            unique_regions = df['Region'].unique()
            
            # Create predictions for each product-region combination
            predictions = []
            
            for product in unique_products:
                for region in unique_regions:
                    # Get latest inventory for this product-region
                    product_region_data = df[(df['Product'] == product) & (df['Region'] == region)]
                    
                    if len(product_region_data) > 0:
                        latest_inventory = product_region_data['Inventory_Level'].iloc[-1]
                        
                        # Create features for prediction
                        future_features = pd.DataFrame({
                            'Year': [future_date.year],
                            'Month': [future_date.month],
                            'Week': [future_date.isocalendar()[1]],
                            'DayOfYear': [future_date.timetuple().tm_yday]
                        })
                        
                        # Predict demand
                        predicted_demand = model.predict(future_features)[0]
                        
                        # Calculate demand gap
                        demand_gap = predicted_demand - latest_inventory
                        
                        # Determine status
                        if demand_gap > 500:
                            status = "Shortage"
                        elif demand_gap < -500:
                            status = "Overstock"
                        else:
                            status = "Balanced"
                        
                        predictions.append({
                            'Product': product,
                            'Region': region,
                            'Inventory_Level': round(latest_inventory, 2),
                            'Predicted_Demand': round(predicted_demand, 2),
                            'Demand_Gap': round(demand_gap, 2),
                            'Supply_Status': status
                        })
            
            # Create DataFrame from predictions
            summary_df = pd.DataFrame(predictions)
            
            # Calculate statistics
            shortage_count = len(summary_df[summary_df['Supply_Status'] == 'Shortage'])
            overstock_count = len(summary_df[summary_df['Supply_Status'] == 'Overstock'])
            balanced_count = len(summary_df[summary_df['Supply_Status'] == 'Balanced'])
            
            # Find product with highest demand
            product_demand = summary_df.groupby('Product')['Predicted_Demand'].sum().sort_values(ascending=False)
            top_demand_product = product_demand.index[0] if len(product_demand) > 0 else "N/A"
            
            # Create trend chart with historical and future prediction
            # Aggregate historical data by date
            historical_trend = df.groupby('Date')['Quantity_Ordered'].sum().reset_index()
            historical_trend.columns = ['Date', 'Total_Demand']
            
            # Add future prediction
            future_total = summary_df['Predicted_Demand'].sum()
            future_row = pd.DataFrame({'Date': [future_date], 'Total_Demand': [future_total]})
            
            # Combine historical and future
            trend_df = pd.concat([historical_trend, future_row], ignore_index=True)
            trend_df = trend_df.sort_values('Date')
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            historical_dates = trend_df[trend_df['Date'] <= datetime.now()]['Date']
            historical_values = trend_df[trend_df['Date'] <= datetime.now()]['Total_Demand']
            
            # Plot future prediction
            future_dates = trend_df[trend_df['Date'] >= historical_dates.iloc[-1]]['Date']
            future_values = trend_df[trend_df['Date'] >= historical_dates.iloc[-1]]['Total_Demand']
            
            ax.plot(historical_dates, historical_values, marker='o', linewidth=2, 
                   markersize=6, label='Historical Demand', color='#667eea')
            ax.plot(future_dates, future_values, marker='o', linewidth=2, 
                   markersize=8, label='Predicted Demand', color='#f5576c', linestyle='--')
            
            ax.axvline(x=future_date, color='red', linestyle=':', linewidth=2, 
                      label='Prediction Date', alpha=0.7)
            
            ax.set_title("Demand Trend Analysis", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Total Demand (Units)", fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"

    return render_template('index_0.html', 
                         summary_table=summary_df, 
                         plot_url=plot_url, 
                         future_date=future_date,
                         error_message=error_message,
                         top_demand_product=top_demand_product,
                         shortage_count=shortage_count,
                         overstock_count=overstock_count,
                         balanced_count=balanced_count)

if __name__ == "__main__":
    app.run(debug=True,port=8000)