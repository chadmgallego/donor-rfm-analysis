-- =============================================================================
-- NAME NORMALIZATION PROCEDURE
-- =============================================================================
-- Purpose: Resolves inconsistent donor names across DONORS_RAW and
--          TRANSACTIONS_RAW by applying a lookup table (NAMENORMALIZATION)
--          and writing clean results to DONORS_NORM and TRANSACTIONS_NORM.
--
-- Strategy: Full reload (TRUNCATE + INSERT). Safe to rerun at any time.
--
-- Lookup table (NAMENORMALIZATION) supports two match strategies:
--   1. Name-only match:  RAW_NAME → CANONICAL_NAME (when RAW_EMAIL IS NULL)
--   2. Email match:      RAW_EMAIL → CANONICAL_NAME (disambiguates same names)
--
-- Priority: Email match wins over name match (via COALESCE order).
--           If neither matches, the original FAMILY_NAME is kept as-is.
--
-- Deduplication: DONORS_NORM uses QUALIFY ROW_NUMBER() to keep one row per
--                FAMILY_NAME (ordered by EMAIL). TRANSACTIONS_NORM does NOT
--                deduplicate — all transactions are preserved.
-- =============================================================================


CREATE OR REPLACE PROCEDURE DONORS.DONORS_SCHEMA.NAME_NORMALIZATION()
  RETURNS STRING
  LANGUAGE SQL
AS
BEGIN

  -- -------------------------------------------------------------------------
  -- STEP 1: Clear target tables (full reload strategy)
  -- -------------------------------------------------------------------------
  TRUNCATE TABLE DONORS.DONORS_SCHEMA.DONORS_NORM;
  TRUNCATE TABLE DONORS.DONORS_SCHEMA.TRANSACTIONS_NORM;

  -- -------------------------------------------------------------------------
  -- STEP 2: Normalize DONORS_RAW → DONORS_NORM
  -- -------------------------------------------------------------------------
  -- Two LEFT JOINs to NAMENORMALIZATION:
  --   nn: matches on FAMILY_NAME where RAW_EMAIL IS NULL (generic name fixes
  --       like typos, household formats). Catches cases where the name alone
  --       is enough to identify the canonical form.
  --   ne: matches on EMAIL (disambiguates donors who share the same name but
  --       have different emails, e.g. two "Amy Jacobs").
  --
  -- COALESCE priority: email match (ne) > name match (nn) > original name
  --
  -- QUALIFY: If JOIN fanout produces multiple rows for the same FAMILY_NAME,
  --          keep only one (ordered by EMAIL for determinism).
  -- -------------------------------------------------------------------------

  INSERT INTO DONORS.DONORS_SCHEMA.DONORS_NORM
  WITH NORM_DONORS AS (
  SELECT 
    d.DONOR_TYPE, 
    COALESCE(ne.CANONICAL_NAME, nn.CANONICAL_NAME, d.FAMILY_NAME) AS FAMILY_NAME,
    d.EMAIL,
    d.ADDRESS_LINE_1,
    d.ADDRESS_LINE_2,
    d.CITY,
    d.STATE,
    d.POSTAL_CODE,
    d.COUNTRY
  FROM DONORS.DONORS_SCHEMA.DONORS_RAW d
  LEFT JOIN DONORS.DONORS_SCHEMA.NAMENORMALIZATION nn
    ON d.FAMILY_NAME = nn.RAW_NAME
    AND nn.RAW_EMAIL IS NULL 
  LEFT JOIN DONORS.DONORS_SCHEMA.NAMENORMALIZATION ne
    ON d.EMAIL = ne.RAW_EMAIL
  )

  SELECT 
    * 
  FROM NORM_DONORS
  QUALIFY ROW_NUMBER() OVER (PARTITION BY FAMILY_NAME ORDER BY EMAIL) = 1;

  -- -------------------------------------------------------------------------
  -- STEP 3: Normalize TRANSACTIONS_RAW → TRANSACTIONS_NORM
  -- -------------------------------------------------------------------------
  -- Same two-JOIN strategy but adapted for transactions:
  --   nn: matches on FAMILY_NAME (generic name fixes, RAW_EMAIL IS NULL)
  --   ne: matches on FAMILY_NAME + CURRENCY_TYPE = PAYMENT_TYPE
  --       (disambiguates donors who share the same name but use different
  --       payment methods)
  --
  -- COALESCE priority: payment-match (ne) > name match (nn) > original name
  --
  -- NOTE: No deduplication here — every transaction row is preserved.
  -- -------------------------------------------------------------------------

  INSERT INTO DONORS.DONORS_SCHEMA.TRANSACTIONS_NORM
  WITH NORM_TRANS AS (
  SELECT 
    t.TRANS_DATE, 
    COALESCE(ne.CANONICAL_NAME, nn.CANONICAL_NAME, t.FAMILY_NAME) AS FAMILY_NAME,
    t.RECURRING,
    t.DESCRIPTION,
    t.AMOUNT,
    t.CURRENCY_TYPE
  FROM DONORS.DONORS_SCHEMA.TRANSACTIONS_RAW t
  LEFT JOIN DONORS.DONORS_SCHEMA.NAMENORMALIZATION nn
    ON t.FAMILY_NAME = nn.RAW_NAME
    AND nn.RAW_EMAIL IS NULL 
  LEFT JOIN DONORS.DONORS_SCHEMA.NAMENORMALIZATION ne
    ON t.FAMILY_NAME = ne.RAW_NAME
    AND t.CURRENCY_TYPE = ne.PAYMENT_TYPE
  )

  SELECT DONORS.DONORS_SCHEMA.NAMENORMALIZATION
    * 
  FROM NORM_TRANS; 

  RETURN 'Normalized data load complete';
END;

-- Execute the procedure
CALL DONORS.DONORS_SCHEMA.NAME_NORMALIZATION();
