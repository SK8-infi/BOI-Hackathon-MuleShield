# Visual Analysis — BOI Mule Account Dataset

14 graphs generated and saved to [files/graphs/](file:///c:/Github/BOI Hackathon/files/graphs).
Each visualization is analyzed below with key takeaways.

---

## 1. Class Imbalance

![Class Imbalance](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/01_class_imbalance.png)

> [!CAUTION]
> **111:1 imbalance** — Only 81 suspicious accounts out of 9,082.
> A naive model predicting "legitimate" for everything would score 99.11% accuracy but catch zero fraud.

**Takeaways:**
- Must use **SMOTE/ADASYN** or **class weights** during training
- Accuracy is a meaningless metric — use **F1-score** and **PR-AUC** instead
- Stratified splitting is mandatory for cross-validation

---

## 2. Missing Values Distribution

![Missing Values](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/02_missing_values.png)

**Takeaways:**
- Left: Most columns with missing data are missing for **10-30%** of rows — a cluster around 10% and another around 26%
- Right: Every single row is missing ~**1,084 features** (27.6%) — this is structural, not random
- The bimodal pattern in column-level missingness suggests **product-dependent features** (e.g., loan features are null for non-loan accounts)

---

## 3. Feature Type Distribution

![Feature Types](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/03_feature_types.png)

**Takeaways:**
- **882 negative-value features** (largest group) — these are delta/change metrics, critical for detecting behavioral shifts
- **527 binary features** — likely product/channel flags
- **478 proportion features** (0-1 range) — behavioral scores and ratios
- **422 features to drop** (359 constant + 63 all-null) — zero predictive value

---

## 4. Categorical Features vs Fraud Rate

![Categorical Fraud Rates](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/04_categorical_fraud_rates.png)

> [!IMPORTANT]
> **Students** have the highest fraud rate at 1.94% — nearly **3x** the dataset average of 0.89%.

**Key findings by category:**
- **Occupation:** Students (1.94%) > Agriculture (1.26%) > Retired (1.04%) > Salaried (0.73%)
- **Branch Area:** Rural (1.44%) is **2.3x** Metro (0.62%)
- **Customer Segment:** RETAIL (1.18%) is **6.2x** CORPORATE (0.19%)
- **Account Type:** Savings (1.28%) dominates — 76 of 81 suspicious accounts
- **Gender:** Males (1.26%) are overrepresented
- **Account Scheme:** L365D (1.26%) slightly above G365D (0.95%); short-term schemes (L7D, L14D, L90D) have **zero** fraud

---

## 5. Key Features: Legit vs Suspicious Distributions

![Key Features Comparison](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/05_key_features_comparison.png)

> [!TIP]
> The dashed vertical lines show the class means. Where the red (suspicious) and teal (legit) dashed lines separate, we have predictive signal.

**Clear separation patterns:**
- **F115 (Risk Score):** Suspicious accounts cluster near 0.8, legit spread 0.3-0.7 — suspicious have visibly higher risk scores
- **F670 (Alert Flag):** Binary — suspicious accounts have the flag set 2.6x more often
- **F2082 (Rare Txn):** Suspicious accounts show almost **zero** rare transaction activity vs. a spread for legit
- **F2122 (Digital):** Suspicious accounts have near-zero digital channel usage — they don't use mobile/online banking
- **F2956 (Txn Count):** Suspicious accounts cluster at very low counts vs. a wide spread for legit
- **F3894 (Age):** Suspicious accounts skew **younger** (peak at 20-35), legit has a broader age spread

**Weak/overlapping patterns:**
- **F2678 (Amt Dev):** Both classes spike near zero — limited visual separation
- **F3836 (Balance):** Extreme skew makes visual comparison difficult, but the sign flip (noted in prior analysis) is a strong signal

---

## 6. Top 20 Correlated Features (F3912 excluded)

![Top Correlations](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/06_top_correlations.png)

**Takeaways:**
- **ALL top 20 have positive correlation** — suspicious accounts have *higher* values across all top features
- **F2506/F2507** are tied at 0.1845 — likely the same metric computed differently (delta/change features)
- **F2408/F2409** are tied at 0.1571 — another duplicate pair
- The max correlation is only **0.18** — no single feature is a strong standalone predictor
- This confirms we need **non-linear models** (tree-based) to capture interactions between weak features
- Interestingly, the 18 features suggested in the problem statement are NOT in this top 20 — the data-driven top features differ from domain suggestions

---

## 7. Feature Block Composition

![Feature Blocks](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/07_feature_blocks.png)

**Takeaways:**
- **F0-499:** Dominated by proportions (yellow) and small floats (blue) — behavioral ratios and scores
- **F500-999:** Heavy in constants (gray) — many inactive product flags
- **F1000-1499:** Binary flags (green) + large floats (purple) — product holdings and monetary amounts
- **F1500-1999:** Large floats dominant — transaction volumes
- **F2000-2499:** Large floats — derived statistical aggregations
- **F2500-2999:** **Almost entirely negative-capable floats (red)** — this is the change/delta block, our most predictive area
- **F3000-3499:** Also heavy in negatives — growth rates and deviation scores
- **F3500-3999:** Mixed — includes metadata, categoricals, and alert flags

---

## 8. Suspicious Account Profile

![Suspicious Profile](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/08_suspicious_profile.png)

**The typical mule account is:**
- A **self-employed (32%)** or **student (28%)** individual
- From a **Rural (36%)** or **Semi-Urban (26%)** branch
- With a **Savings account (94%)**
- This maps to the real-world pattern: young individuals in smaller towns are recruited as mule account holders

---

## 9. Data Leakage — Critical Finding

![Data Leakage](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/09_data_leakage.png)

> [!CAUTION]
> **Two features MUST be excluded from all models:**

**Left — F3912 (Fraud Flag):**
- The heatmap shows 79 of 82 F3912=1 accounts are also F3924=1 (suspicious)
- Only 5 mismatches across the entire dataset
- Including this would create a model that simply copies this flag

**Right — F2230 (Time Period):**
- **100%** of non-Oct25 accounts are suspicious (48+23+10 = 81)
- **0%** of Oct25 accounts are suspicious
- The temporal split is perfect — a trivial rule achieves 100% accuracy
- This is clearly a data collection artifact, not a real pattern

---

## 10. Key Feature Box Plots

![Box Plots](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/10_key_feature_boxplots.png)

**Takeaways:**
- **F115 (Risk):** Median for suspicious is noticeably higher — clear upward shift
- **F321 (Velocity):** Similar medians but legit has wider spread — suspicious accounts have uniform behavior
- **F527 (Peer):** Nearly identical — weak standalone predictor
- **F531 (Diversity):** Legit accounts show wider whiskers — suspicious accounts lack transaction diversity
- **F1692 (Activity):** Suspicious has a lower median — less account activity
- **F2122 (Digital):** Strikingly different — suspicious median is at ~0, legit has visible spread upward

---

## 11. Key Feature Correlation Heatmap

![Correlation Heatmap](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/11_correlation_heatmap.png)

**Notable correlations:**
- **F2956 ↔ F3043 = 0.98** — near-duplicate features (likely total transaction counts over different periods)
- **F321 ↔ F531 = 0.50** — velocity and diversity are moderately linked
- **F1692 ↔ F2122 = 0.33** — activity count and digital usage are weakly linked
- **F3887 ↔ F3894 = 0.40** — account age and customer age have moderate positive correlation
- **Most features are near-zero correlated with each other** — they capture independent signals, which is excellent for model performance (low multicollinearity)

> [!TIP]
> The near-zero inter-correlations mean each feature contributes unique information — a tree-based model will benefit from having access to all of them.

---

## 12. Age Distributions

![Age Distributions](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/12_age_distributions.png)

**Account Age (F3887):**
- Both classes are heavily right-skewed — most accounts are **< 6 months old**
- Suspicious accounts are concentrated in the **0-3 month range** — newly opened accounts
- This confirms mule accounts are freshly created and used quickly

**Customer Age (F3894):**
- Suspicious accounts peak sharply at **25-35 years** 
- Legitimate accounts have a broader, flatter distribution extending to 70+
- Younger demographic is more likely to be recruited for mule activity

---

## 13. Block-Level Importance

![Block Importance](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/13_block_importance.png)

*(Available at [files/graphs/13_block_importance.png](file:///c:/Github/BOI Hackathon/files/graphs/13_block_importance.png))*

Shows mean and max absolute correlation with target by 500-feature block.

---

## 14. Missing Values by Class

![Missing by Class](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/14_missing_by_class.png)

**Takeaways:**
- Legit accounts: mean **1,084** missing features
- Suspicious accounts: mean **1,120** missing features — slightly higher
- The distributions are very similar — missingness is **not a strong class signal**
- However, the suspicious distribution has a slightly heavier right tail
- Adding a **"row_missing_count" feature** could provide marginal signal

---

## Summary of Key Visual Insights

| Finding | Visual Evidence | Impact on Model |
|---------|----------------|-----------------|
| 111:1 class imbalance | Graph 1 | Must use SMOTE/class weights |
| F3912 data leakage | Graph 9 (left) | DROP — 0.97 correlation with target |
| F2230 temporal leakage | Graph 9 (right) | DROP — 100%/0% temporal split |
| Students = highest risk | Graph 4 | Occupation is a strong categorical feature |
| Rural = highest risk area | Graph 4 | Branch area adds predictive value |
| Suspicious = low digital | Graph 5, 10 | F2122 is one of the best features |
| Suspicious = newer accounts | Graph 12 | Account age (F3887) is predictive |
| Delta features = most predictive | Graph 6, 7 | F2500-F2999 block is the gold mine |
| Features are mostly independent | Graph 11 | Low multicollinearity = good for trees |
| Missingness is structural | Graph 2, 14 | Not class-dependent, use tree models |

> [!NOTE]
> All 14 graphs are saved permanently in [files/graphs/](file:///c:/Github/BOI Hackathon/files/graphs) for presentation and documentation purposes.
