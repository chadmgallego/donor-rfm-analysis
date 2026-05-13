import pandas as pd
import numpy as np

# ----------------------------------------------------------------
# BUILD QUANTILE BINS FOR FREQUENCY & MONETARY (MONTHLY DONORS)
# ----------------------------------------------------------------

def compute_bins(df, feature):
    # Compute quartile-based bins for a feature
    # Returns: [0, 25%, 50%, 75%, inf]
    q1, q2, q3 = df[feature].quantile([0.25, 0.50, 0.75])
    return [0, q1, q2, q3, np.inf]

def model(dbt, session):
    donors_df = dbt.ref('donor_behavioral_features').to_pandas()
    # ----------------------------------------------------------------
    # INITIAL SEGMENT LABELING BASED ON RECURRING STATUS
    # ----------------------------------------------------------------

    # Start with no segment label
    donors_df['SEGMENT'] = None

    # One-time donors
    one_time_mask = ~donors_df['IS_RECURRING']
    donors_df.loc[one_time_mask, 'SEGMENT'] = 'one_time'

    # Recurring but not monthly
    rec_not_monthly_mask = donors_df['IS_RECURRING'] & ~donors_df['IS_MONTHLY']
    donors_df.loc[rec_not_monthly_mask, 'SEGMENT'] = 'rec_not_monthly'

    # Keep a mask and subset for monthly donors
    monthly_mask = donors_df['IS_MONTHLY']
    monthly_df = donors_df.loc[monthly_mask]

    # ----------------------------------------------------------------
    # R-SCORE (RECENCY) FOR MONTHLY DONORS
    # Rule: recency relative to 1.5 * median donation interval
    # ----------------------------------------------------------------

    donors_df['R_SCORE'] = None # initialize as blank

    interval = donors_df['MEDIAN_RECENT_INTERVAL']
    recency = donors_df['RECENCY']

    # Classify fully engaged monthly donors (not lapsed or churned)
    donors_df.loc[monthly_mask & (recency < 1.5*interval), 'R_SCORE'] = 3

    # Classify monthly donors who have lapsed in their giving
    donors_df.loc[monthly_mask & (recency >= 1.5*interval) & (recency <= 120), 'R_SCORE'] = 2

    # Classify monthly donors who have churned (considered to be lost)
    donors_df.loc[monthly_mask & (recency >= 120), 'R_SCORE'] = 1

    # Compute bins for monthly donors (monthly_df)
    frequency_bins = compute_bins(monthly_df, 'FREQUENCY')
    monetary_bins = compute_bins(monthly_df, 'MONETARY')

    # ----------------------------------------------------------------
    # ASSIGN F-SCORE & M-SCORE USING QUARTILE BINS
    # ----------------------------------------------------------------

    # Initialize scores 
    donors_df[['F_SCORE', 'M_SCORE']] = None

    # Compute frequency scores
    donors_df.loc[monthly_mask, 'F_SCORE'] = pd.cut(monthly_df['FREQUENCY'], bins = frequency_bins, labels = [1, 2, 3, 4], include_lowest = True)

    # Compute monetary scores
    donors_df.loc[monthly_mask, 'M_SCORE'] = pd.cut(monthly_df['MONETARY'], bins = monetary_bins, labels = [1, 2, 3, 4], include_lowest = True)

    # ----------------------------------------------------------------
    # MERGE RFM SCORES FOR MONTHLY DONORS TO MAIN DATAFRAME
    # ----------------------------------------------------------------

    # Reassign monthly_df with mask 
    monthly_df = donors_df.loc[monthly_mask]

    donors_df['RFM_SCORE'] = monthly_df.assign(
        RFM_SCORE = lambda df: df['R_SCORE'].astype(str) + df['F_SCORE'].astype(str) + df['M_SCORE'].astype(str)
    )['RFM_SCORE']


    # Define a segmentation map using regular expressions and RFM scores
    seg_map = {
        r'2[1-4][1-4]': 'lapsed', # Monthly donors who have lapsed in their giving but not fully churned -- 1.5x median_recent_interval < recency < 120 
        r'1[1-4][1-4]': 'churned', # Monthly donors who have churned (considered to be lost or fully unengaged) -- 120 < recency 
        r'3[1-2]4': 'big_spenders', # Monthly donors who are engaged & give large amounts but have not yet given frequently -- potential champions
        r'3[3-4][1-3]': 'loyal_donors', # Monthly donors who are engaged, have given the most frequently, but not as generously as champions
        r'[3]44': 'champions' # Monthly donors who are engaged, have given the most frequently and most monetary -- most valuable donors
    }

    # Reassign monthly_df with mask 
    monthly_df = donors_df.loc[monthly_mask]

    # Segment donors according to seg_map, adding easily identifiable lables (e.g., 'lapsed')
    donors_df.loc[monthly_mask, 'SEGMENT'] = monthly_df['RFM_SCORE'].replace(seg_map, regex = True)

    # All donors who do not fall into the aforementioned categories will be labeled as 'other'
    donors_df.loc[~donors_df['SEGMENT'].isin(list(seg_map.values()) + ['rec_not_monthly', 'one_time']), 'SEGMENT'] = 'other'
    
    return donors_df