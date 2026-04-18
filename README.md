# Retail Demand Forecast & Inventory Assistant

**A simple and practical demand forecasting tool designed specifically for small independent retail shops.**

This dashboard helps small shop owners predict future demand, decide how much stock to reorder, and understand their store’s unique sales patterns without needing any technical knowledge.

### What This Tool Does
- Predicts future sales for the next 7 to 60 days
- Suggests how much stock you should reorder
- Adjusts safety stock based on your risk preference (Conservative / Neutral / Aggressive)
- Shows your store’s unique patterns 
- Works even if your data is incomplete

### How to Use This Tool 

1. **Download the files**
   - Click the green **Code** button → **Download ZIP**
   - Unzip the folder on your computer

2. **Install Python (only once)**
   - Go to https://www.python.org/downloads/
   - Download and install the latest Python version (recommended: 3.10 or 3.11)
   - During installation, **check the box** "Add python.exe to PATH"

3. **Install required libraries**
   - Open **Command Prompt** (Windows) or **Terminal** (Mac)
   - Navigate to the project folder using `cd` command
   - Run this command:  pip install -r requirements.txt
  
4. **Run the dashboard**
   - In the same terminal, run: streamlit run dashboard.py
   - Your web browser will automatically open the dashboard

5. **How to use the dashboard**
  - Click **Browse files** and upload your sales CSV file
  - Adjust the settings on the top (Risk Level, Cash Flow Cycle, Lead Time, Forecast Horizon)
  - Click on the tabs to see forecasts and business insights
  - Download the result as CSV if needed

**Important**: 
  - Your sales file must have at least these columns: `Date`, `Product ID`, `Units Sold`, `Price`
  - The more complete your data is, the better the insights will be

### Author
Developed by Chau Chu Hei as a final year project (CI601).
