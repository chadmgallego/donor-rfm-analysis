-- =============================================================================
-- DUPLICATES CHECK TABLE & PROCEDURE
-- Purpose: Identify potential duplicate donor records based on matching
--          FAMILY_NAME, EMAIL, or ADDRESS_LINE_1 for human-in-the-loop review.
-- =============================================================================

-- Create a staging table to hold flagged duplicate records.
-- Mirrors the structure of DONORS_NORM so rows can be inserted directly.
CREATE OR REPLACE TABLE DONORS.DONORS_SCHEMA.DONORS_DUPLICATES (
    DONOR_TYPE VARCHAR,
    FAMILY_NAME VARCHAR,
    EMAIL VARCHAR,
    ADDRESS_LINE_1 VARCHAR,
    ADDRESS_LINE_2 VARCHAR,
    CITY VARCHAR,
    STATE VARCHAR,
    POSTAL_CODE VARCHAR,
    COUNTRY VARCHAR
);

-- Procedure: LOAD_DUPLICATES
-- Truncates the duplicates table and re-populates it with all rows from
-- DONORS_NORM where ANY of the following fields appear more than once:
--   1. FAMILY_NAME  – catches same-name donors (possible family or re-entry)
--   2. EMAIL        – catches shared/reused email addresses
--   3. ADDRESS_LINE_1 – catches same physical address
-- A row only needs to match ONE condition to be included (OR logic).
CREATE OR REPLACE PROCEDURE DONORS.DONORS_SCHEMA.LOAD_DUPLICATES()
  RETURNS STRING
  LANGUAGE SQL
AS
BEGIN
  -- Clear previous results so the table always reflects the current state
  TRUNCATE TABLE DONORS.DONORS_SCHEMA.DONORS_DUPLICATES;

  -- Insert all rows that share a FAMILY_NAME, EMAIL, or ADDRESS_LINE_1
  -- with at least one other row (i.e., the value appears 2+ times).
  INSERT INTO DONORS.DONORS_SCHEMA.DONORS_DUPLICATES
  SELECT *
  FROM DONORS.DONORS_SCHEMA.DONORS_NORM
  WHERE FAMILY_NAME IN (
    SELECT FAMILY_NAME
    FROM DONORS.DONORS_SCHEMA.DONORS_NORM
    GROUP BY FAMILY_NAME
    HAVING COUNT(*) > 1
    )
    
  OR EMAIL IN (
    SELECT EMAIL
    FROM DONORS.DONORS_SCHEMA.DONORS_NORM
    GROUP BY EMAIL
    HAVING COUNT(*) > 1
    )
  OR ADDRESS_LINE_1 IN (
    SELECT ADDRESS_LINE_1
    FROM DONORS.DONORS_SCHEMA.DONORS_NORM
    GROUP BY ADDRESS_LINE_1
    HAVING COUNT(*) > 1
  );

  -- Return a confirmation message for the caller / task log
  RETURN 'Loaded possible duplicates into DONORS_DUPLICATES for HITL';
END;

-- Execute the procedure to populate the duplicates table now
CALL DONORS.DONORS_SCHEMA.LOAD_DUPLICATES();

-- Review flagged duplicates, ordered by name for easier manual comparison
SELECT 
    * 
FROM DONORS.DONORS_SCHEMA.DONORS_DUPLICATES
ORDER BY FAMILY_NAME;