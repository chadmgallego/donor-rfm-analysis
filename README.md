# 🎗️ Donor Behavior Analytics & Revenue Stability Assessment

A full end-to-end data analytics project covering data cleaning, deduplication, feature engineering, RFM segmentation, and interactive Tableau dashboards — applied to the donor base of Tokyo Student Mobilization, Inc. Built to support strategic retention, re-engagement, and geographically informed fundraising decisions.

> **v2 Update:** The analysis pipeline has been fully migrated from Jupyter notebooks to a production-grade data stack using Snowflake and dbt. Raw data is ingested and normalized in Snowflake, transformed via dbt models (SQL + Python), and served to Tableau for visualization.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Context & Impact](#business-context--impact)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [dbt Model Lineage](#dbt-model-lineage)
- [Key Metrics & Findings](#key-metrics--findings)
- [Donor Segment Overview](#donor-segment-overview)
- [Geographic Analysis](#geographic-analysis)
- [Visualizations](#visualizations)
- [Limitations & Future Work](#limitations--future-work)
- [Repository Structure](#repository-structure)

---

## Project Overview

This project cleans, engineers, and segments a multi-year donor dataset from Student Mobilization, Inc., covering **139 donors** and **2,384 transactions** spanning February 2022 through December 2025. The pipeline resolves duplicate donor identities, constructs behavioral features from raw transaction histories, and applies a custom RFM (Recency-Frequency-Monetary) scoring framework to classify donors into actionable segments.

### Version History

| Version | Stack | Description |
|---|---|---|
| v1 | Python, Jupyter, pandas | All analysis in sequential notebooks; CSV-based pipeline |
| v2 | Snowflake, dbt, Python (Snowpark) | Fully migrated to cloud warehouse + dbt transformation layer |

**v2 dbt models:**

| Model | Type | Description |
|---|---|---|
| `donor_base_metrics` | SQL | Base aggregations per donor from raw transactions |
| `donor_behavioral_features` | Python | Monthly donor classification, median donation interval |
| `donor_segmentation_rfm` | Python | RFM scoring, quantile binning, segment assignment |

---

## Business Context & Impact

Tokyo Student Mobilization operates with a lean team and conducts annual in-person fundraising visits to the United States. With limited time on the ground, strategic targeting is essential. This project addresses that need directly:

- **Establishes a revenue baseline** — $313,106.88 in total donations across 4 years, with a median gift of $100.00 and a 31.07% monthly donor churn rate, giving leadership a clear picture of revenue health and risk
- **Quantifies donor value by segment** — identifies that Champions (12% of donors) account for 30% of total donation volume, while Churned donors (23% of the base) represent 18% of volume and the highest-potential recovery opportunity
- **Delivers a donor health framework** — RFM scoring distinguishes 8 behavioral segments, enabling targeted outreach rather than blanket communication
- **Surfaces a high-value re-engagement opportunity** — 22% of donors have given only once (~30 individuals), with several one-time gifts exceeding $1,000; structured outreach to this group could yield significant revenue recovery
- **Informs geographic strategy** — Arizona and Texas account for the largest share of both donors and donation volume, providing clear prioritization criteria for annual U.S. visits
- **Reveals monthly donor churn risk** — 32 of 103 monthly donors (31%) are classified as churned based on recency, highlighting the need for a proactive retention cadence

---

## Tech Stack

| Category | Tools |
|---|---|
| **Cloud Data Warehouse** | Snowflake |
| **Data Ingestion & Normalization** | Snowflake Stages, Streams, Tasks, Stored Procedures |
| **Transformation Layer** | dbt (SQL + Python models via Snowpark) |
| **Data Processing** | Python, `pandas`, `numpy` (via dbt Python models) |
| **Deduplication & Identity Resolution** | Custom Snowflake stored procedures (email pattern matching, address-based household consolidation) |
| **Feature Engineering** | dbt SQL aggregations + pandas (groupby, `.diff()`, `.transform()`) |
| **EDA & Visualization (v1)** | `matplotlib`, `seaborn` |
| **Statistics** | `scipy.stats` — IQR outlier detection, normal approximation, confidence intervals |
| **Segmentation** | Custom RFM scoring (quantile binning + regex segment mapping) |
| **Reporting** | Excel (pivot tables, segment summary), Tableau (interactive dashboard) |

---

## Project Architecture

```
Raw CSV Exports (donors, transactions)
      │
      ▼
Snowflake Ingestion Layer
  ├── External stages (S3/local CSV load)
  ├── Streams & Tasks (incremental change capture)
  ├── Normalization stored procedure (name/email resolution)
  └── Deduplication procedure (household consolidation)
      │
      ▼
dbt Transformation Layer
  │
  ├── donor_base_metrics.sql
  │     ├── Cast TRANS_DATE string → TIMESTAMP_TZ
  │     ├── Cast AMOUNT string → FLOAT
  │     ├── Aggregate per donor: SUM, COUNT, MIN, MAX, AVG, MEDIAN
  │     ├── Compute RECENCY (days since last donation via DATEDIFF)
  │     └── Derive IS_RECURRING (FREQUENCY > 1)
  │
  ├── donor_behavioral_features.py  (Snowpark / pandas)
  │     ├── Pull donor_base_metrics + raw transactions
  │     ├── Cast TRANS_DATE → datetime with UTC
  │     ├── Sort transactions per donor ascending
  │     ├── Compute DAYS_SINCE_PREV via groupby diff()
  │     ├── Derive MEDIAN_RECENT_INTERVAL (tail(3) median per donor)
  │     └── Derive IS_MONTHLY (interval < 35 days & IS_RECURRING)
  │
  └── donor_segmentation_rfm.py  (Snowpark / pandas)
        ├── Initial segment labeling (one_time, rec_not_monthly)
        ├── R-score: recency vs. 1.5× median donation interval
        ├── Quantile bins for FREQUENCY & MONETARY (25/50/75th pct)
        ├── F-score & M-score via pd.cut (quartile labels 1–4)
        ├── RFM score concatenation (R + F + M string)
        └── Regex-based segment mapping:
              ├── champions      → RFM: 344
              ├── loyal_donors   → RFM: 3[3-4][1-3]
              ├── big_spenders   → RFM: 3[1-2]4
              ├── lapsed         → RFM: 2[1-4][1-4]
              └── churned        → RFM: 1[1-4][1-4]
      │
      ├──► reports/DonorContributionOverviewPublic.xlsx
      │      ├── ANONYMIZED_DONOR_TABLE    # Full cleaned donor export (no names)
      │      └── CONTRIBUTION_OVERVIEW     # Segment-level pivot summary
      │
      └──► Tableau: Donor Behavioral Trends & Segmentation Overview
             ├── KPI tiles (total donors, monthly donors, total donations, median gift, churn rate)
             ├── Monthly donation trend (area chart, Feb 2022 – Dec 2025)
             ├── Geographic map (donor density by state)
             ├── Segment count + donation volume bar charts
             └── Interactive date and segment filters
```

---

## dbt Model Lineage

```
sources (Snowflake)
  ├── donors_schema.donors_norm
  └── donors_schema.transactions_norm
        │
        ▼
  donor_base_metrics        [SQL table]
        │
        ▼
  donor_behavioral_features [Python table]
        │
        ▼
  donor_segmentation_rfm    [Python table]
```

---

## Key Metrics & Findings

### Revenue & Donor Base Summary
- **Total donors analyzed:** 139
- **Monthly donors:** 109 (103 recurring monthly + 6 recurring not monthly)
- **One-time donors:** 30 (~22% of donor base)
- **Total donations:** $313,106.88
- **Median donation:** $100.00
- **Monthly donor churn rate:** 31.07% (32 of 103 monthly donors classified as churned)

### Donation Distribution
- **Median donation amount:** $100.00 — also the mode, representing the most common recurring gift tier
- **Distribution shape:** Right-skewed on a log scale, with a dominant peak at $100 and a secondary cluster near $50; a long upper tail reflects infrequent but high-value gifts
- **Outlier proportion:** 16.5% of donors have made at least one outlier donation (95% CI: 0.103 – 0.227)

### Monthly Donor Breakdown

| Category | Count |
|---|---|
| Recurring Monthly | 103 |
| Recurring Not Monthly | 6 |
| One Time | 30 |
| **Total** | **139** |

### RFM Feature Distributions (Monthly Donors)

| Feature | Median | IQR | Notable |
|---|---|---|---|
| `recency` (days) | ~30 | 0 – 330 | Right-skewed; most donors recently active |
| `frequency` (donations) | ~18 | 15 – 35 | Middle 50% of monthly donors cluster in high-frequency range |
| `monetary` (total USD) | ~$2,000 | $1,000 – $3,000 | Symmetric with outliers up to $14,234 |

**Notable finding:** The recency box plot reveals that the median, lower quartile, and minimum are tightly clustered — indicating that the majority of monthly donors remain recently engaged, with churn concentrated in a smaller tail of lapsed supporters.

---

## Donor Segment Overview

Full segment-level statistics are available in [`reports/DonorContributionOverviewPublic.xlsx`](reports/DonorContributionOverviewPublic.xlsx). Key highlights below.

- **Champions** (17 donors) — fully engaged, highest frequency and monetary value; median gift of ~$119, accounting for ~30% of total donation volume. Retaining this segment is the single highest fundraising priority.
- **Churned** (32 donors) — 23% of the donor base with a median gift of ~$91; account for ~18% of total volume. Direct personal outreach to this group represents the highest-potential revenue recovery opportunity.
- **Other** (31 donors) — the second-largest segment by count; median gift ~$98. The high concentration of Texas donors here suggests the current segmentation may not fully capture behavioral diversity in that state.
- **Loyal Donors** (16 donors) — frequently engaged monthly donors with moderate giving; median gift ~$61. Strong candidates for targeted appreciation campaigns to reinforce engagement.
- **One Time** (30 donors) — 22% of the base, with a median gift of ~$1,076 driven by several high-value single gifts (up to $10,000). High-capacity donors with untapped recurring potential.
- **Big Spenders** (4 donors) — fully engaged donors with large gifts but lower frequency; average median gift of $445. Potential pipeline to Champions status with targeted cultivation.
- **Lapsed** (3 donors) — monthly donors whose cadence has slowed but who have not fully churned; median gift ~$110. Small group, but high conversion probability with personalized outreach.
- **Rec Not Monthly** (6 donors) — recurring donors with irregular giving intervals; average median gift of $634. May respond well to giving cadence nudges or annual campaign framing.

**Segment contribution summary:**

| Segment | Donors | Avg Median Gift | Min | Max |
|---|---|---|---|---|
| big_spenders | 4 | $445.00 | $250 | $515 |
| champions | 17 | $118.63 | $51.50 | $250 |
| churned | 32 | $91.22 | $10.30 | $257.50 |
| lapsed | 3 | $109.58 | $100 | $128.75 |
| loyal_donors | 16 | $61.04 | $15.45 | $100 |
| one_time | 30 | $1,075.61 | $12 | $10,000 |
| other | 31 | $98.15 | $10.30 | $257.50 |
| rec_not_monthly | 6 | $634.33 | $51.50 | $1,500 |

---

## Geographic Analysis

- **Arizona (AZ)** — the dominant donor hub with 65+ donors, the highest total donations (~$167,000), and the largest count of Champions (11). Also has the most one-time and churned donors, presenting both a retention priority and a re-engagement opportunity in a single geography.
- **Texas (TX)** — second-highest in donor count (35+) and total donations (~$80,000), but dominated by the `other` segment (24 donors). Signals a need for deeper exploratory analysis or enhanced segmentation to better characterize giving behavior in this state.
- **Colorado (CO)** — third-largest donor base with 12 donors and ~$29,000 in total donations. Relatively balanced across segments, indicating a moderate but stable donor community.

**Strategic implication:** Annual U.S. fundraising visits should prioritize Arizona for both Champion retention events and lapsed/churned reactivation outreach, followed by Texas for broad-based engagement and segment clarification.

---

## Visualizations

### Probability Distribution of Donations & Recurring vs. One-Time Donors
![donation_patterns](figures/donation_patterns.png)
> Left: Log-transformed distribution of all donation amounts, with a median of $100 — both the most common recurring gift and the distribution mode. The long right tail reflects infrequent but high-impact gifts. Right: 78% of donors have given more than once, underscoring a strong base of recurring supporters.

### RFM Feature Distributions (Monthly Donors)
![rfm_boxplots](figures/rfm_features_box.png)
> Box plots for recency, frequency, and monetary value across the 103 monthly donors. Recency is highly right-skewed — median near zero — confirming that most monthly donors remain actively engaged.

### Total Donations & Donor Counts by State
![state_analysis](figures/state_by_state_bar.png)
> Arizona and Texas lead in both total donation volume and donor concentration.

### Donor Segment Distribution Across States
![segment_heatmap](figures/segment_heatmap.png)
> Heatmap of donor segment counts for Arizona, Texas, and Colorado.

### Number of Donors & Total Donations Per Segment
![segment_bars](figures/donor_segmentation.png)
> Champions generate the largest share of donation volume despite representing only 12% of donors.

### Median Donation Distributions by Segment
![segment_kde](figures/segment_median_amounts.png)
> KDE plot of median donation amounts per segment on a log scale.

### Tableau Dashboard: Donor Behavioral Trends & Segmentation Overview
![tableau_dashboard](dashboards/Donor_Trends_Segmentation_Overview.png)
> Interactive dashboard summarizing 4 years of donor activity with KPI tiles, time-series trend, geographic map, and segment bar charts.

---

## Limitations & Future Work

**Current limitations:**
- **Incomplete donor coverage** — this dataset represents one subdivision of the broader Student Mobilization donor base
- **Limited donor attributes** — no demographic, campaign-specific, or engagement data were available
- **Quantile binning artifacts** — RFM scores depend on quartile thresholds which can misclassify donors near boundaries
- **Static snapshot** — recency calculations are anchored to the model run date

**Planned improvements:**
- Expand dataset to include other organizational subdivisions
- Incorporate campaign-level and engagement data (email, events)
- Explore unsupervised clustering (k-means, DBSCAN) as an alternative to quantile-based RFM scoring
- Schedule dbt runs for automated data refresh
- Build a predictive churn model to identify at-risk monthly donors before lapsing occurs

---

## Repository Structure

```
donor-rfm-analysis/
├── dashboards/
│   └── Donor_Trends_Segmentation_Overview.png
├── data/
│   ├── donors/
│   │   ├── 01_donor_base_metrics_anonymized.csv
│   │   ├── 02_donor_behavioral_features_anonymized.csv
│   │   └── 03_donor_segmentation_rfm_anonymized.csv
│   └── transactions/
│       ├── 01_transactions_cleaned_anonymized.csv
│       └── 02_transactions_processed_anonymized.csv
├── figures/
│   ├── donation_patterns.png
│   ├── donor_segmentation.png
│   ├── rfm_features_box.png
│   ├── segment_heatmap.png
│   ├── segment_median_amounts.png
│   └── state_by_state_bar.png
├── models/
│   └── donors/
│       ├── donor_base_metrics.sql         # Base donor aggregations
│       ├── donor_behavioral_features.py   # Monthly classification + interval features
│       ├── donor_segmentation_rfm.py      # RFM scoring + segment assignment
│       ├── schema.yml                     # Column types, tests, descriptions
│       └── sources.yml                    # Snowflake source definitions
├── notebooks/                             # v1 analysis (Jupyter)
│   ├── 01_data_cleaning_and_metrics.ipynb
│   ├── 02_exploratory_analysis_and_feature_engineering.ipynb
│   ├── 03_donor_segmentation_and_overview.ipynb
│   └── 04_insights_recommendations_and_limitations.ipynb
├── reports/
│   └── DonorContributionOverviewPublic.xlsx
├── snowflake/                             # Snowflake infrastructure SQL (pre-dbt layer)
│   ├── donors_db_schema_tables.sql        # Database, schema, and table definitions
│   ├── donors_stages_streams_tasks.sql    # Ingestion pipeline (stages, streams, tasks)
│   ├── duplicates_check_table_procedure.sql
│   ├── manual_refresh.sql
│   ├── name_normalization_table.sql
│   └── normalization_procedure.sql
├── dbt_project.yml
└── README.md
```

---

*Data sourced from internal donor records provided by Student Mobilization, Inc. All analysis is performed on anonymized or pseudonymized data. Raw donor files are not publicly shared to protect individual privacy.*
