# 09 — Data Leakage Analysis

⚠️ **CRITICAL — READ BEFORE ANY MODELING WORK** ⚠️

Two features in the dataset exhibit data leakage — they contain information 
that would not be available at prediction time and must be handled carefully.

---

## 🔴 F3912 — Fraud/Mule Alert Flag (MUST EXCLUDE)

### Evidence

| Metric | Value |
|--------|-------|
| Data type | int64 |
| Unique values | 2 ({0, 1}) |
| Correlation with target | **0.9691** |
| Distribution | 0: 9,000 / 1: 82 |

### Cross-tabulation with Target (F3924)

|  | F3924 = 0 (Legit) | F3924 = 1 (Suspicious) |
|---|---|---|
| **F3912 = 0** | 8,998 | 2 |
| **F3912 = 1** | 3 | 79 |

### Interpretation
- F3912 = 1 for 82 accounts, of which **79 are suspicious** (96.3% precision)
- F3912 = 0 for 9,000 accounts, of which **only 2 are suspicious** (0.02% miss rate)
- This is almost certainly a **fraud investigation flag** or **alert outcome** — it was likely 
  set AFTER the account was identified as suspicious, making it a downstream artifact of the target.
- The 3 false positives (F3912=1, F3924=0) and 2 false negatives (F3912=0, F3924=1) represent 
  minor label disagreements between the flag and the target.

### Decision: **EXCLUDE FROM ALL MODELS**
Including F3912 would create a model that simply copies this flag, achieving ~97% recall 
but providing zero predictive value for new/unseen accounts.

---

## 🟡 F2230 — Temporal Period (HANDLE WITH CAUTION)

### Evidence

| Value | Total Accounts | Suspicious | Suspicious Rate |
|-------|---------------|-----------|----------------|
| Oct25 | 9,001 | 0 | **0.00%** |
| Sep25 | 48 | 48 | **100.00%** |
| Nov25 | 23 | 23 | **100.00%** |
| Dec25 | 10 | 10 | **100.00%** |

### Interpretation
- **ALL 81 suspicious accounts** come from Sep25, Nov25, or Dec25
- **ALL 9,001 Oct25 accounts** are legitimate
- This creates a **perfect temporal separator** — a simple rule `if period != Oct25 → suspicious` 
  achieves 100% recall and 100% precision on this dataset

### Why This Is Problematic
1. The model would learn to simply check the time period instead of actual behavioral patterns
2. In production, new data would come from a different time period, making this rule useless
3. The suspicious accounts were likely identified through a separate investigation process 
   and added to the dataset from different time batches

### Decision Options
1. **EXCLUDE F2230** from features (safest approach)
2. **Use with awareness**: If the time period genuinely captures seasonal fraud patterns, 
   it may have legitimate predictive value — but this seems unlikely given the 100%/0% split
3. **Use for stratified splitting**: Ensure train/test splits don't separate by time period

### Recommended Decision: **EXCLUDE FROM FEATURES**

---

## 🟢 Neighboring Alert Flags (F3908–F3914)

These are in the same block as F3912 and should be reviewed:

| Feature | Unique | Distribution | Corr Target | Risk Level |
|---------|--------|-------------|-------------|------------|
| F3908 | 2 | 0:6300, 1:2782 | +0.0970 | Low — common flag (30.6%), weak correlation |
| F3909 | 2 | 0:9031, 1:51 | +0.0399 | Low — rare flag (0.56%) |
| F3910 | 2 | 0:8842, 1:240 | -0.0083 | Safe — negligible correlation |
| F3911 | 1 | 0:9082 | N/A | Safe — constant (no info) |
| **F3912** | **2** | **0:9000, 1:82** | **+0.9691** | **🔴 LEAKAGE** |
| F3913 | 2 | 1:5234, 0:3848 | -0.0632 | Low — common flag |
| F3914 | 2 | 0:6185, 1:2897 | -0.0549 | Low — common flag |

**Assessment**: Only F3912 is a clear leakage risk. F3908–F3914 (excluding F3912) appear 
to be legitimate binary features (e.g., product flags, verification flags) with weak 
correlations — they can likely be retained.

---

## Summary of Exclusions

| Feature | Action | Reason |
|---------|--------|--------|
| F3912 | **DROP** | Near-perfect correlation (0.97) with target — fraud flag leakage |
| F2230 | **DROP** | Perfect temporal separator — 100%/0% split by period |
| Unnamed: 0 | **DROP** | Row index, not a feature |
| 63 all-null columns | **DROP** | Zero information |
| 359 constant columns | **DROP** | Zero information |
