# Retail Demand Forecast & Inventory Assistant

**A lightweight, practical demand forecasting tool designed specifically for small independent retail shops.**

This dashboard helps small shop owners predict future demand, plan stock levels, and understand their store’s unique sales patterns without requiring technical knowledge.

## What This Tool Does
- **Forecasts future sales** for the next **7–60 days**
- **Recommends reorder quantities** based on predicted demand
- **Calculates safety stock** using your chosen risk level  
  *(Conservative / Neutral / Aggressive)*
- **Identifies store‑specific patterns**, including:
  - Weekly sales rhythm  
  - Promotion effects  
  - Weather impact  
  - Seasonality

This tool is built for the realities of small retailers: messy data, limited time, and the need for simple, actionable insights.
---
### 1. Installation (One-time setup)

1. **Download the files**
   - Click the green **Code** button → **Download ZIP**
   - Unzip the folder on your computer

2. **Install Python**
   - Download from: https://www.python.org/downloads/
   - Recommended: **Python 3.10 or 3.11**

3. **Install required libraries**
   - Open **Command Prompt** (Windows) or **Terminal** (Mac)
   - Navigate to the project folder
   - Run: `pip install -r requirements.txt`
  
4. **Run the dashboard**
   - In the same terminal, Run: `streamlit run dashboard.py`
   - Your web browser will automatically open the dashboard

### 2. Preparing Your Sales CSV File

**Required columns** (must have these):
- `Date` → format: YYYY-MM-DD (e.g. 2025-01-15)
- `Product ID`
- `Units Sold`
- `Price`

**Optional columns** (recommended — improves accuracy and insights):
- `Category`
- `Inventory Level`
- `Weather Condition`
- `Holiday/Promotion` (1 = yes, 0 = no)
- `Seasonality`
- `Units Ordered`
- `Discount`

**Tip**: The more complete your data is, the better the forecasts and business insights will be.  
Even if some optional columns are missing, the dashboard will still work and clearly tell you what is affected.

### Author
Developed by Chau Chu Hei as a final year project (CI601).
