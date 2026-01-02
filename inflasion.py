from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

# ------------------------------
# Load dataset and train model
# ------------------------------
df = pd.read_csv("fintech_app_usage_dataset_inflation.csv")

# Simulate a 'Month' column if not present
if "Month" not in df.columns:
    df["Month"] = np.random.choice(range(1, 13), len(df))

df['Date'] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
df = df.sort_values('Date')

# Aggregate inflation by month
df['Year_Month'] = df['Date'].dt.to_period('M')
monthly_inflation = df.groupby('Year_Month')['Inflation_Rate(%)'].mean().reset_index()
monthly_inflation['Date'] = monthly_inflation['Year_Month'].dt.to_timestamp()
monthly_inflation['Year'] = monthly_inflation['Date'].dt.year
monthly_inflation['Month'] = monthly_inflation['Date'].dt.month
monthly_inflation['MonthIndex'] = range(len(monthly_inflation))

# Train model
X = monthly_inflation[['Year', 'Month', 'MonthIndex']]
y = monthly_inflation['Inflation_Rate(%)']
model = LinearRegression()
model.fit(X, y)

# Calculate average inflation
avg_inflation = monthly_inflation['Inflation_Rate(%)'].mean()

# ------------------------------
# Flask routes
# ------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    prediction_month = None
    predicted_inflation = None
    inflation_status = None
    error_message = None
    avg_inflation_display = None
    deviation = None

    if request.method == 'POST':
        prediction_date_str = request.form.get('prediction_date', None)
        
        if not prediction_date_str:
            error_message = "Please select a prediction month."
            return render_template('index1.html',
                                 plot_url=plot_url,
                                 prediction_month=prediction_month,
                                 predicted_inflation=predicted_inflation,
                                 inflation_status=inflation_status,
                                 error_message=error_message)
        
        try:
            prediction_date = pd.to_datetime(prediction_date_str + "-01")
            prediction_month = prediction_date.strftime("%B %Y")
            
            # Calculate month index
            first_date = monthly_inflation['Date'].min()
            months_diff = (prediction_date.year - first_date.year) * 12 + (prediction_date.month - first_date.month)
            
            # Predict inflation
            future_data = pd.DataFrame({
                'Year': [prediction_date.year],
                'Month': [prediction_date.month],
                'MonthIndex': [months_diff]
            })
            
            predicted_inflation = model.predict(future_data)[0]
            
            # Determine inflation status
            if predicted_inflation > avg_inflation + 1:
                inflation_status = "⚠️ HIGH INFLATION"
                status_color = 'red'
                box_color = '#ff6b6b'
            elif predicted_inflation < avg_inflation - 1:
                inflation_status = "✅ LOW INFLATION"
                status_color = 'green'
                box_color = '#51cf66'
            else:
                inflation_status = "➡️ MODERATE INFLATION"
                status_color = 'orange'
                box_color = '#ffa94d'
            
            avg_inflation_display = round(avg_inflation, 2)
            deviation = round(predicted_inflation - avg_inflation, 2)
            
            # Create chart
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot historical data
            ax.plot(monthly_inflation['Date'], monthly_inflation['Inflation_Rate(%)'],
                   label="Historical Inflation", color='#3b5bdb', 
                   linewidth=2.5, alpha=0.9)
            
            # Plot prediction point
            ax.scatter(prediction_date, predicted_inflation,
                      color=status_color, s=400, label=f"Predicted ({prediction_month})",
                      edgecolors='black', linewidths=3, zorder=10, marker='o')
            
            # Glow effect
            ax.scatter(prediction_date, predicted_inflation,
                      color=status_color, s=800, alpha=0.3, zorder=9)
            
            # Average line
            ax.axhline(y=avg_inflation, color='gray', linestyle='--',
                      label=f'Average ({avg_inflation:.1f}%)', alpha=0.6, linewidth=1.5)
            
            # Annotation
            bbox_props = dict(boxstyle='round,pad=1', facecolor=box_color,
                            edgecolor='black', linewidth=2.5, alpha=0.95)
            
            annotation_text = f'{predicted_inflation:.2f}%\n{inflation_status}'
            ax.annotate(annotation_text,
                       xy=(prediction_date, predicted_inflation),
                       xytext=(30, 30), textcoords='offset points',
                       bbox=bbox_props,
                       fontsize=14, fontweight='bold', color='white',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                     color='black', lw=3))
            
            # Formatting
            ax.set_xlabel("Date", fontsize=14, fontweight='bold')
            ax.set_ylabel("Inflation Rate (%)", fontsize=14, fontweight='bold')
            ax.set_title("Inflation Trend with Prediction",
                        fontsize=18, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=12, framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
    
    return render_template('index_1.html',
                         plot_url=plot_url,
                         prediction_month=prediction_month,
                         predicted_inflation=predicted_inflation,
                         inflation_status=inflation_status,
                         error_message=error_message,
                         avg_inflation=avg_inflation_display,
                         deviation=deviation)

if __name__ == "__main__":
    app.run(debug=True, port = 3000)