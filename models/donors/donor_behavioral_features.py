
import pandas as pd 

def median_interv_calc(donors_df, transactions_df):
    
    donors_df = donors_df.merge(
        transactions_df
            # Cast string to datetime with UTC to enable date arithmetic
            .assign(TRANS_DATE=lambda df: pd.to_datetime(df['TRANS_DATE'], utc=True))
            # Sort ascending so diff() computes forward-looking intervals per donor
            .sort_values(by=['FAMILY_NAME', 'TRANS_DATE'], ascending=True)
            # Compute days between consecutive donations per donor
            # diff() on a sorted group gives NaN for first donation (no previous)
            .assign(DAYS_SINCE_PREV=lambda df: df.groupby('FAMILY_NAME')['TRANS_DATE'].diff().dt.days)
            # Reduce to one row per donor — median of last 3 gaps captures recent giving cadence
            # tail(3) limits to most recent intervals to avoid stale history skewing the result
            .groupby('FAMILY_NAME')['DAYS_SINCE_PREV']
            .apply(lambda x: x.tail(3).median())
            .reset_index(name='MEDIAN_RECENT_INTERVAL'),
        how='left',
        on='FAMILY_NAME'  # left join preserves all donors even if no transaction history
    )

    return donors_df


def model(dbt, session): 

    # Pull aggregated donor metrics from upstream SQL model
    donors_df = dbt.ref('donor_base_metrics').to_pandas()

    # Pull raw transactions directly from source to compute per-donation intervals
    # donor_base_metrics only has aggregates, so we need the raw rows here
    transactions_df = dbt.source('donors_schema', 'transactions_norm').to_pandas()

    donors_df = median_interv_calc(donors_df, transactions_df)

    # Flag donors whose median recent donation gap is under 35 days and have donated more than once
    # 35 day threshold accounts for slight variance in monthly giving schedules
    donors_df['IS_MONTHLY'] = (donors_df['MEDIAN_RECENT_INTERVAL'].lt(35) & donors_df['IS_RECURRING'])

    return donors_df