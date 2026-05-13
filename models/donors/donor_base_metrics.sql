-- Cast raw transaction data types before aggregation
WITH TRANSACTIONS_SRC AS (
    SELECT 
        -- ISO 8601 format with timezone offset (+09:00 JST)
        TO_TIMESTAMP_TZ(TRANS_DATE, 'YYYY-MM-DD"T"HH24:MI:SS+TZH:TZM') as TRANS_DATE,
        FAMILY_NAME,
        -- Source amount is string, cast to float to support decimal donations
        CAST(AMOUNT AS FLOAT) AS AMOUNT
    FROM {{ source('donors_schema', 'transactions_norm') }}
),

-- Aggregate transactions to one row per donor family
TRANSACTIONS_AGG AS ( 
    SELECT 
        FAMILY_NAME,
        SUM(AMOUNT) AS MONETARY,           -- total lifetime giving
        COUNT(AMOUNT) AS FREQUENCY,         -- total number of donations
        MIN(TRANS_DATE) AS DONATION_START_DATE, -- date of first donation
        MAX(TRANS_DATE) AS LAST_DONATION_DATE,  -- date of most recent donation
        ROUND(AVG(AMOUNT), 2) AS MEAN_AMOUNT,
        MEDIAN(AMOUNT) AS MED_AMOUNT,
        MAX(AMOUNT) AS MAX_AMOUNT,
        MIN(AMOUNT) AS MIN_AMOUNT,

        -- TRUE if donor has given more than once
        CASE 
            WHEN FREQUENCY > 1 THEN TRUE 
            ELSE FALSE 
        END AS IS_RECURRING,
        
        -- Days since last donation as of model run time
        DATEDIFF('day', LAST_DONATION_DATE, CURRENT_TIMESTAMP()) as RECENCY
    FROM TRANSACTIONS_SRC
    GROUP BY FAMILY_NAME 
)

-- Join donor profile info onto aggregated transaction metrics
-- RIGHT JOIN ensures all donors with transactions are included
-- even if they have no matching record in donors_norm
SELECT 
    d.*,
    t.* EXCLUDE(FAMILY_NAME)
FROM {{ source('donors_schema', 'donors_norm') }} d
RIGHT JOIN TRANSACTIONS_AGG t 
    ON d.FAMILY_NAME = t.FAMILY_NAME

