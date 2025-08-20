# Donor RFM Analysis

This project explores donor giving behavior for Tokyo Student Mobilization using Recency-Frequency-Monetary (RFM) segmentation. The goal is to uncover insights that can support donor engagement, retention, and geographic fundraising efforts.

## Key Questions
- Who are the most consistent and generous donors?
- How many donors are recurring vs. one-time?
- What states have the highest donor activity?
- Are there donors who are at risk of lapsing?

## Methods Used
- Python (pandas, matplotlib, seaborn)
- Data cleaning and transformation
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

