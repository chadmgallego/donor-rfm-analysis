-- =============================================================================
-- MANUAL REFRESH SCRIPT
-- =============================================================================
-- PURPOSE: After uploading new CSV files to internal stages via Snowsight UI,
--          you MUST manually refresh the stage metadata. Internal stages do NOT
--          auto-detect new files — the directory table (which streams monitor)
--          only updates after an explicit REFRESH command.
--
-- WHEN TO RUN: Every time you upload new CSVs through the Snowsight UI.
--              Without this step, streams won't fire and the task DAG won't trigger.
--
-- PIPELINE FLOW AFTER REFRESH:
--   1. REFRESH registers files in the directory table
--   2. Streams detect new directory table entries → SYSTEM$STREAM_HAS_DATA = TRUE
--   3. LOAD_RAW_TASK fires (1-min schedule) → calls LOAD_RAW_FROM_STAGES()
--   4. NORMALIZE_NAMES_TASK fires (child) → calls NAME_NORMALIZATION()
--   5. LOAD_DUPLICATES_TASK fires (child) → calls LOAD_DUPLICATES()
--
-- NOTE: The LOAD_RAW_TASK WHEN clause uses AND — BOTH stages must have new
--       files for the task to fire. Upload to both stages before refreshing.
-- =============================================================================


-- STEP 1: Register newly uploaded files in the stage directory tables.
-- This is what makes the streams aware that new data arrived.
ALTER STAGE DONORS.DONORS_SCHEMA.DONORS_STAGE REFRESH;
ALTER STAGE DONORS.DONORS_SCHEMA.TRANSACTIONS_STAGE REFRESH;


-- STEP 2: Verify row counts across all pipeline tables.
-- Run these AFTER the task DAG completes (~1-2 minutes after refresh).
-- Expected: RAW tables reflect the CSV contents; NORM tables may differ
-- due to deduplication logic in the normalization procedure.
SELECT COUNT(*) FROM DONORS.DONORS_SCHEMA.DONORS_RAW;        -- Should match CSV row count
SELECT COUNT(*) FROM DONORS.DONORS_SCHEMA.DONORS_NORM;       -- Deduplicated by FAMILY_NAME
SELECT COUNT(*) FROM DONORS.DONORS_SCHEMA.TRANSACTIONS_RAW;  -- Should match CSV row count
SELECT COUNT(*) FROM DONORS.DONORS_SCHEMA.TRANSACTIONS_NORM; -- All transactions preserved (no dedup)
SELECT COUNT(*) FROM DONORS.DONORS_SCHEMA.DONORS_DUPLICATES; -- Flagged duplicates from normalization


-- IMPORTANT: In a task DAG, child tasks must be resumed BEFORE the root task.
-- Resume order: children first, then root.
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_DUPLICATES_TASK RESUME;
ALTER TASK DONORS.DONORS_SCHEMA.NORMALIZE_NAMES_TASK RESUME;
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_RAW_TASK RESUME;

-- STEP 4: (OPTIONAL) Clear stage files after confirming data loaded correctly.
-- REMOVE deletes files from the stage but keeps the stage object intact.
-- Do this ONLY after verifying counts above are correct.
-- WARNING: Once removed, you cannot re-run COPY INTO from these files —
--          you would need to re-upload the CSVs.
REMOVE @DONORS.DONORS_SCHEMA.DONORS_STAGE;
REMOVE @DONORS.DONORS_SCHEMA.TRANSACTIONS_STAGE;

-- CHECK TO MAKE SURE FILES WERE DROPPED FROM STAGE
LIST @DONORS.DONORS_SCHEMA.DONORS_STAGE;
LIST @DONORS.DONORS_SCHEMA.TRANSACTIONS_STAGE;
