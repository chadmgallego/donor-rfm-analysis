-- =============================================================================
-- DONORS PIPELINE: Stages, Streams, and Tasks
-- =============================================================================
-- This file sets up an automated ingestion pipeline that detects new CSV uploads
-- to internal stages and loads them into RAW tables.
--
-- LIFECYCLE:
--   1. Upload CSVs to DONORS_STAGE and TRANSACTIONS_STAGE
--   2. Run ALTER STAGE ... REFRESH on each stage (registers files in directory table)
--   3. Streams detect the new file registrations
--   4. Task fires ONLY when BOTH streams have data (AND condition)
--   5. Procedure truncates RAW tables and COPYs fresh data from stages
--
-- IMPORTANT:
--   - Internal stages require manual REFRESH after file uploads
--   - The task will NOT fire unless BOTH stages have new files
--   - TRUNCATE + COPY means RAW tables always reflect the full stage contents
-- =============================================================================


-- -----------------------------------------------------------------------------
-- STAGES
-- -----------------------------------------------------------------------------
-- DIRECTORY = (ENABLE = TRUE) creates a directory table on each stage.
-- A directory table is a metadata layer that tracks files on the stage
-- (names, sizes, timestamps). Streams attach to this directory table —
-- not the raw files — to detect when new files appear.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE STAGE DONORS.DONORS_SCHEMA.DONORS_STAGE
  DIRECTORY = (ENABLE = TRUE)
  FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"');

CREATE OR REPLACE STAGE DONORS.DONORS_SCHEMA.TRANSACTIONS_STAGE
  DIRECTORY = (ENABLE = TRUE)
  FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"');


-- -----------------------------------------------------------------------------
-- STREAMS (on stages)
-- -----------------------------------------------------------------------------
-- A stream on a stage tracks CDC (change data capture) against the directory table.
-- When new files are registered (via ALTER STAGE ... REFRESH), the stream records
-- those additions. SYSTEM$STREAM_HAS_DATA() returns TRUE when unconsumed
-- records exist in the stream.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE STREAM DONORS.DONORS_SCHEMA.DONORS_STAGE_STREAM
  ON STAGE DONORS.DONORS_SCHEMA.DONORS_STAGE;

CREATE OR REPLACE STREAM DONORS.DONORS_SCHEMA.TRANSACTIONS_STAGE_STREAM
  ON STAGE DONORS.DONORS_SCHEMA.TRANSACTIONS_STAGE;


-- -----------------------------------------------------------------------------
-- STORED PROCEDURE
-- -----------------------------------------------------------------------------
-- A task body can only be a single SQL statement. Since we need multiple steps
-- (TRUNCATE + COPY + COPY), we wrap them in a procedure.
--
-- Logic:
--   1. Truncate both RAW tables (full reload strategy — not incremental)
--   2. COPY all files from each stage into the corresponding RAW table
--   3. Return a confirmation string
-- -----------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE DONORS.DONORS_SCHEMA.LOAD_RAW_FROM_STAGES()
  RETURNS STRING
  LANGUAGE SQL
AS
BEGIN
  TRUNCATE TABLE DONORS.DONORS_SCHEMA.DONORS_RAW;
  TRUNCATE TABLE DONORS.DONORS_SCHEMA.TRANSACTIONS_RAW;

  COPY INTO DONORS.DONORS_SCHEMA.DONORS_RAW
    FROM @DONORS.DONORS_SCHEMA.DONORS_STAGE
    FORCE = TRUE;

  COPY INTO DONORS.DONORS_SCHEMA.TRANSACTIONS_RAW
    FROM @DONORS.DONORS_SCHEMA.TRANSACTIONS_STAGE
    FORCE = TRUE;
  
  RETURN 'Load complete';
END;


-- -----------------------------------------------------------------------------
-- TASK
-- -----------------------------------------------------------------------------
-- The task polls on a default 1-minute schedule (no explicit SCHEDULE set).
-- WHEN clause: uses AND so the task ONLY executes when BOTH streams have
-- unconsumed data. If only one stage has new files, the task waits.
--
-- After the task runs successfully, both streams are consumed (reset),
-- and the task goes back to polling until both stages have new files again.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE TASK DONORS.DONORS_SCHEMA.LOAD_RAW_TASK
  WAREHOUSE = COMPUTE_WH
  SCHEDULE = '1 MINUTE'
  WHEN
    SYSTEM$STREAM_HAS_DATA('DONORS.DONORS_SCHEMA.DONORS_STAGE_STREAM')
    AND SYSTEM$STREAM_HAS_DATA('DONORS.DONORS_SCHEMA.TRANSACTIONS_STAGE_STREAM')
AS
  CALL DONORS.DONORS_SCHEMA.LOAD_RAW_FROM_STAGES();

-- -----------------------------------------------------------------------------
-- CHILD TASK (DAG): Normalize names after raw data loads
-- -----------------------------------------------------------------------------
-- AFTER = LOAD_RAW_TASK makes this a child task in a DAG (directed acyclic graph).
-- It automatically runs when LOAD_RAW_TASK completes successfully.
-- No WHEN clause needed — the parent's success is the trigger.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE TASK DONORS.DONORS_SCHEMA.NORMALIZE_NAMES_TASK
  WAREHOUSE = COMPUTE_WH
  AFTER DONORS.DONORS_SCHEMA.LOAD_RAW_TASK
AS
  CALL DONORS.DONORS_SCHEMA.NAME_NORMALIZATION();

-- -----------------------------------------------------------------------------
-- CHILD TASK (DAG): Load duplicates after normalization completes
-- -----------------------------------------------------------------------------
-- AFTER = NORMALIZE_NAMES_TASK makes this the next step in the DAG.
-- It calls the LOAD_DUPLICATES procedure to flag potential duplicate donors.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE TASK DONORS.DONORS_SCHEMA.LOAD_DUPLICATES_TASK
  WAREHOUSE = COMPUTE_WH
  AFTER DONORS.DONORS_SCHEMA.NORMALIZE_NAMES_TASK
AS
  CALL DONORS.DONORS_SCHEMA.LOAD_DUPLICATES();


-- Check recent task runs for your DAG
SELECT NAME, STATE, SCHEDULED_TIME, COMPLETED_TIME, ERROR_MESSAGE
FROM TABLE(DONORS.INFORMATION_SCHEMA.TASK_HISTORY(
  SCHEDULED_TIME_RANGE_START => DATEADD('HOUR', -1, CURRENT_TIMESTAMP()),
  RESULT_LIMIT => 20
))
WHERE NAME IN ('LOAD_RAW_TASK', 'NORMALIZE_NAMES_TASK', 'LOAD_DUPLICATES_TASK')
ORDER BY SCHEDULED_TIME DESC;



-- =============================================================================
-- OPTIONAL: CONVERT TASK DAG TO SERVERLESS TRIGGERED TASKS
-- =============================================================================
-- PURPOSE: Eliminates the 1-minute polling schedule that keeps COMPUTE_WH running
--          24/7 (even when no data arrives). Instead, tasks fire ONLY when streams
--          detect new data — zero cost when idle.
--
-- WHY SERVERLESS TRIGGERED?
--   - Current setup: COMPUTE_WH polls every 1 min → never auto-suspends → ~30 credits/day
--   - Serverless triggered: no schedule, no warehouse — Snowflake spins up compute
--     ONLY when SYSTEM$STREAM_HAS_DATA returns TRUE, then shuts down immediately
--   - Per-credit cost is ~1.5x higher, but total cost is drastically lower because
--     you only pay for actual execution time (seconds), not idle warehouse hours
--
-- REQUIREMENTS:
--   - The ACCOUNTADMIN role (or a role with EXECUTE MANAGED TASK) is needed
--   - TARGET_COMPLETION_INTERVAL is required for serverless triggered tasks
--     (tells Snowflake how to size compute — it will auto-scale to finish in time)
--
-- IMPORTANT: Suspend child tasks BEFORE modifying the root task.
--            Resume in reverse order: children first, then root.
--
-- HOW TO USE: Run these statements in order after you're ready to re-enable
--             the pipeline in serverless mode.
-- =============================================================================

-- STEP 1: Make sure all tasks are suspended before modifying
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_DUPLICATES_TASK SUSPEND;
ALTER TASK DONORS.DONORS_SCHEMA.NORMALIZE_NAMES_TASK SUSPEND;
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_RAW_TASK SUSPEND;

-- STEP 2: Convert the ROOT task to serverless triggered mode
-- UNSET SCHEDULE: removes the 1-minute polling — task becomes event-driven
-- REMOVE WAREHOUSE: detaches COMPUTE_WH so Snowflake manages compute
-- TARGET_COMPLETION_INTERVAL: Snowflake will size resources to finish within 5 min
-- The existing WHEN clause (SYSTEM$STREAM_HAS_DATA) is preserved automatically —
-- it becomes the trigger condition instead of a skip-if-false guard
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_RAW_TASK UNSET SCHEDULE;
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_RAW_TASK REMOVE WAREHOUSE;
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_RAW_TASK SET TARGET_COMPLETION_INTERVAL = '5 MINUTE';

-- STEP 3: Convert CHILD tasks to serverless as well
-- Child tasks inherit the trigger from the root (they fire after root completes),
-- so they don't need SCHEDULE or WHEN clauses — just remove the warehouse
ALTER TASK DONORS.DONORS_SCHEMA.NORMALIZE_NAMES_TASK REMOVE WAREHOUSE;
ALTER TASK DONORS.DONORS_SCHEMA.NORMALIZE_NAMES_TASK SET TARGET_COMPLETION_INTERVAL = '5 MINUTE';

ALTER TASK DONORS.DONORS_SCHEMA.LOAD_DUPLICATES_TASK REMOVE WAREHOUSE;
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_DUPLICATES_TASK SET TARGET_COMPLETION_INTERVAL = '5 MINUTE';

-- STEP 4: Resume tasks (children FIRST, then root)
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_DUPLICATES_TASK RESUME;
ALTER TASK DONORS.DONORS_SCHEMA.NORMALIZE_NAMES_TASK RESUME;
ALTER TASK DONORS.DONORS_SCHEMA.LOAD_RAW_TASK RESUME;

-- STEP 5: Verify the conversion — SCHEDULE should be NULL, WAREHOUSE should be empty
SHOW TASKS IN SCHEMA DONORS.DONORS_SCHEMA;