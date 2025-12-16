# Donor Analysis Using RFM Segmentation

This project explores donor giving behavior for Tokyo Student Mobilization using Recency-Frequency-Monetary (RFM) segmentation. The goal is to uncover insights that can support donor engagement, retention, and geographic fundraising efforts.

## Key Questions
- What percentage of donors are recurring? 
- Do most monthly donors have fairly symmetric giving patterns (little to no extreme outliers in their donation amounts)?
- How is the donor base distributed across different states?
- Who are the most consistent and generous donors?
- Are there donors who have lapsed or fully churned and may benefit from re-engagement efforts?

## Methods Used
- Python (pandas, matplotlib, seaborn)
- Data cleaning and transformation
- Exploratory Data Analysis (EDA)
- Quantile-based binning for RFM scoring
- Donor segmentation into groups (Champions, At-Risk, One-Time, etc.)

## Key Insights
- 23% of donors are one-time contributors — potential for re-engagement.
- "Champions" make up the majority of total donation volume.
- Arizona and Texas account for the highest concentration of donor activity.
- Recency is heavily right-skewed — most donors gave recently.

## Data Source Disclosure
Due to the sensitive nature of donor information, the raw CSV files are not publicly shared. All analyses, visualizations, and summaries are based on internal datasets to protect individual privacy. 

## Files
- `donor_rfm_analysis.ipynb`: Full Jupyter notebook with code, visualizations, and insights.

