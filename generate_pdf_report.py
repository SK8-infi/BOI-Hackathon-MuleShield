"""
Generate a PDF report with embedded graphs from the visual analysis.
Output: files/Graph_Analysis_Report.pdf
"""
from fpdf import FPDF
import os

GRAPHS_DIR = 'files/graphs'
OUT_PDF = 'files/Graph_Analysis_Report.pdf'


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, 'BOI Hackathon - Visual Analysis Report', align='L')
            self.cell(0, 8, f'Page {self.page_no()}', align='R', new_x="LMARGIN", new_y="NEXT")
            self.line(10, 16, 200, 16)
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, 'PSBs Cybersecurity, Fraud & AI Hackathon 2026 | Bank of India x IIT Hyderabad', align='C')

    def add_title_page(self):
        self.add_page()
        self.ln(50)
        self.set_font('Helvetica', 'B', 28)
        self.set_text_color(20, 60, 120)
        self.cell(0, 15, 'Visual Analysis Report', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        self.set_font('Helvetica', '', 16)
        self.set_text_color(60, 60, 60)
        self.cell(0, 10, 'BOI Mule Account Classification', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 10, 'Problem Statement 2', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(15)
        self.set_draw_color(20, 60, 120)
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)
        self.set_font('Helvetica', '', 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, 'Dataset: 9,082 accounts | 3,925 features | 81 suspicious', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, '14 Visualizations with Analysis', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(20)
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "PSBs Cybersecurity, Fraud & AI Hackathon 2026", align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, 'Bank of India in collaboration with IIT Hyderabad', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, 'Powered by DFS (Ministry of Finance) / IBA', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, 'June 2026', align='C', new_x="LMARGIN", new_y="NEXT")

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(20, 60, 120)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def alert_box(self, text, alert_type='NOTE'):
        colors = {
            'CAUTION': (180, 30, 30),
            'IMPORTANT': (180, 120, 0),
            'TIP': (0, 120, 80),
            'NOTE': (40, 80, 160),
        }
        r, g, b = colors.get(alert_type, (40, 80, 160))
        self.set_fill_color(r, g, b)
        self.rect(10, self.get_y(), 3, 14, 'F')
        bg_colors = {
            'CAUTION': (255, 240, 240),
            'IMPORTANT': (255, 250, 230),
            'TIP': (230, 250, 240),
            'NOTE': (230, 240, 255),
        }
        br, bg, bb = bg_colors.get(alert_type, (230, 240, 255))
        self.set_fill_color(br, bg, bb)
        self.set_x(14)
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(r, g, b)
        # Measure text height
        line_count = len(text) // 85 + 1
        box_h = max(14, line_count * 5.5 + 6)
        self.rect(13, self.get_y(), 187, box_h, 'F')
        self.set_x(16)
        self.multi_cell(181, 5.5, f"  {alert_type}: {text}")
        self.ln(3)

    def add_image_page(self, img_path, width=185):
        if os.path.exists(img_path):
            # Check if there's enough space, otherwise new page
            if self.get_y() + 90 > 270:
                self.add_page()
            self.image(img_path, x=12, w=width)
            self.ln(5)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.set_x(x + 5)
        # Split bold markers
        self.cell(5, 5.5, '-', new_x="END")
        self.set_x(self.get_x() + 2)
        # Handle **bold** in text
        parts = text.split('**')
        for i, part in enumerate(parts):
            if i % 2 == 1:
                self.set_font('Helvetica', 'B', 10)
            else:
                self.set_font('Helvetica', '', 10)
            self.write(5.5, part)
        self.ln(6)

    def add_summary_table(self):
        headers = ['Finding', 'Visual Evidence', 'Impact on Model']
        col_widths = [60, 35, 95]
        
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align='C')
        self.ln()
        
        rows = [
            ('111:1 class imbalance', 'Graph 1', 'Must use SMOTE/class weights'),
            ('F3912 data leakage', 'Graph 9 (left)', 'DROP - 0.97 corr with target'),
            ('F2230 temporal leakage', 'Graph 9 (right)', 'DROP - 100%/0% temporal split'),
            ('Students = highest risk', 'Graph 4', 'Occupation is a strong feature'),
            ('Rural = highest risk area', 'Graph 4', 'Branch area adds predictive value'),
            ('Suspicious = low digital', 'Graph 5, 10', 'F2122 is one of the best features'),
            ('Suspicious = newer accts', 'Graph 12', 'Account age (F3887) is predictive'),
            ('Delta features = best', 'Graph 6, 7', 'F2500-F2999 block is the gold mine'),
            ('Features mostly independent', 'Graph 11', 'Low multicollinearity = good for trees'),
            ('Missingness is structural', 'Graph 2, 14', 'Not class-dependent, use tree models'),
        ]
        
        self.set_font('Helvetica', '', 8)
        self.set_text_color(40, 40, 40)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 7, val, border=1, fill=True, align='L')
            self.ln()
            fill = not fill


def build_report():
    pdf = ReportPDF()

    # ── Title Page ──
    pdf.add_title_page()

    # ── Section 1: Class Imbalance ──
    pdf.add_page()
    pdf.section_title('1. Class Imbalance')
    pdf.add_image_page(f'{GRAPHS_DIR}/01_class_imbalance.png')
    pdf.alert_box(
        '111:1 imbalance - Only 81 suspicious accounts out of 9,082. '
        'A naive model predicting "legitimate" for everything would score 99.11% accuracy but catch zero fraud.',
        'CAUTION'
    )
    pdf.bullet_point('Must use **SMOTE/ADASYN** or **class weights** during training')
    pdf.bullet_point('Accuracy is a meaningless metric - use **F1-score** and **PR-AUC** instead')
    pdf.bullet_point('Stratified splitting is mandatory for cross-validation')

    # ── Section 2: Missing Values ──
    pdf.add_page()
    pdf.section_title('2. Missing Values Distribution')
    pdf.add_image_page(f'{GRAPHS_DIR}/02_missing_values.png')
    pdf.bullet_point('Left: Most columns with missing data are missing for **10-30%** of rows')
    pdf.bullet_point('Right: Every row is missing ~**1,084 features** (27.6%) - this is structural, not random')
    pdf.bullet_point('Bimodal pattern suggests **product-dependent features** (e.g., loan features null for non-loan accounts)')

    # ── Section 3: Feature Types ──
    pdf.add_page()
    pdf.section_title('3. Feature Type Distribution')
    pdf.add_image_page(f'{GRAPHS_DIR}/03_feature_types.png')
    pdf.bullet_point('**882 negative-value features** (largest group) - delta/change metrics, critical for detecting behavioral shifts')
    pdf.bullet_point('**527 binary features** - likely product/channel flags')
    pdf.bullet_point('**478 proportion features** (0-1 range) - behavioral scores and ratios')
    pdf.bullet_point('**422 features to drop** (359 constant + 63 all-null) - zero predictive value')

    # ── Section 4: Categorical Fraud Rates ──
    pdf.add_page()
    pdf.section_title('4. Categorical Features vs Fraud Rate')
    pdf.add_image_page(f'{GRAPHS_DIR}/04_categorical_fraud_rates.png')
    pdf.alert_box(
        'Students have the highest fraud rate at 1.94% - nearly 3x the dataset average of 0.89%.',
        'IMPORTANT'
    )
    pdf.bullet_point('**Occupation:** Students (1.94%) > Agriculture (1.26%) > Retired (1.04%) > Salaried (0.73%)')
    pdf.bullet_point('**Branch Area:** Rural (1.44%) is **2.3x** Metro (0.62%)')
    pdf.bullet_point('**Customer Segment:** RETAIL (1.18%) is **6.2x** CORPORATE (0.19%)')
    pdf.bullet_point('**Account Type:** Savings (1.28%) dominates - 76 of 81 suspicious accounts')
    pdf.bullet_point('**Gender:** Males (1.26%) are overrepresented')

    # ── Section 5: Key Features Comparison ──
    pdf.add_page()
    pdf.section_title('5. Key Features: Legit vs Suspicious')
    pdf.add_image_page(f'{GRAPHS_DIR}/05_key_features_comparison.png')
    pdf.alert_box(
        'Dashed vertical lines show class means. Where the red (suspicious) and teal (legit) lines separate, we have signal.',
        'TIP'
    )
    pdf.bullet_point('**F115 (Risk Score):** Suspicious cluster near 0.8, legit spread 0.3-0.7')
    pdf.bullet_point('**F670 (Alert Flag):** Binary - suspicious have the flag set 2.6x more often')
    pdf.bullet_point('**F2082 (Rare Txn):** Suspicious show almost **zero** rare transaction activity')
    pdf.bullet_point('**F2122 (Digital):** Suspicious have near-zero digital channel usage')
    pdf.bullet_point('**F2956 (Txn Count):** Suspicious cluster at very low counts')
    pdf.bullet_point('**F3894 (Age):** Suspicious skew **younger** (peak at 20-35)')

    # ── Section 6: Top Correlations ──
    pdf.add_page()
    pdf.section_title('6. Top 20 Correlated Features (F3912 excluded)')
    pdf.add_image_page(f'{GRAPHS_DIR}/06_top_correlations.png')
    pdf.bullet_point('**ALL top 20 have positive correlation** - suspicious accounts have higher values')
    pdf.bullet_point('**F2506/F2507** tied at 0.1845 - likely same metric computed differently')
    pdf.bullet_point('Max correlation is only **0.18** - no single feature is a strong standalone predictor')
    pdf.bullet_point('Confirms we need **non-linear models** (tree-based) to capture interactions')
    pdf.bullet_point('The 18 features from problem statement are NOT in top 20 - data-driven features differ')

    # ── Section 7: Feature Blocks ──
    pdf.add_page()
    pdf.section_title('7. Feature Block Composition')
    pdf.add_image_page(f'{GRAPHS_DIR}/07_feature_blocks.png')
    pdf.bullet_point('**F0-499:** Proportions (yellow) and small floats (blue) - behavioral ratios')
    pdf.bullet_point('**F500-999:** Heavy in constants (gray) - many inactive product flags')
    pdf.bullet_point('**F1000-1499:** Binary flags + large floats - product holdings and amounts')
    pdf.bullet_point('**F2500-2999:** Almost entirely **negative-capable floats** - the change/delta block, most predictive')
    pdf.bullet_point('**F3500-3999:** Mixed - metadata, categoricals, and alert flags')

    # ── Section 8: Suspicious Profile ──
    pdf.add_page()
    pdf.section_title('8. Suspicious Account Profile')
    pdf.add_image_page(f'{GRAPHS_DIR}/08_suspicious_profile.png')
    pdf.body_text('The typical mule account is:')
    pdf.bullet_point('A **self-employed (32%)** or **student (28%)** individual')
    pdf.bullet_point('From a **Rural (36%)** or **Semi-Urban (26%)** branch')
    pdf.bullet_point('With a **Savings account (94%)**')
    pdf.bullet_point('Maps to real-world pattern: young individuals in smaller towns recruited as mule account holders')

    # ── Section 9: Data Leakage ──
    pdf.add_page()
    pdf.section_title('9. Data Leakage - Critical Finding')
    pdf.add_image_page(f'{GRAPHS_DIR}/09_data_leakage.png')
    pdf.alert_box('Two features MUST be excluded from all models: F3912 and F2230.', 'CAUTION')
    pdf.body_text(
        'F3912 (Fraud Flag): The heatmap shows 79 of 82 F3912=1 accounts are also F3924=1. '
        'Only 5 mismatches across the entire dataset. Including this would create a model that simply copies this flag.'
    )
    pdf.body_text(
        'F2230 (Time Period): 100% of non-Oct25 accounts are suspicious (48+23+10 = 81). '
        '0% of Oct25 accounts are suspicious. This is a data collection artifact, not a real pattern.'
    )

    # ── Section 10: Box Plots ──
    pdf.add_page()
    pdf.section_title('10. Key Feature Box Plots')
    pdf.add_image_page(f'{GRAPHS_DIR}/10_key_feature_boxplots.png')
    pdf.bullet_point('**F115 (Risk):** Median for suspicious is noticeably higher - clear upward shift')
    pdf.bullet_point('**F321 (Velocity):** Similar medians but legit has wider spread')
    pdf.bullet_point('**F531 (Diversity):** Legit accounts show wider whiskers - suspicious lack diversity')
    pdf.bullet_point('**F2122 (Digital):** Strikingly different - suspicious median is at ~0')

    # ── Section 11: Correlation Heatmap ──
    pdf.add_page()
    pdf.section_title('11. Key Feature Correlation Heatmap')
    pdf.add_image_page(f'{GRAPHS_DIR}/11_correlation_heatmap.png')
    pdf.bullet_point('**F2956 / F3043 = 0.98** - near-duplicate features')
    pdf.bullet_point('**F321 / F531 = 0.50** - velocity and diversity are moderately linked')
    pdf.bullet_point('**F3887 / F3894 = 0.40** - account age and customer age correlated')
    pdf.alert_box(
        'Most features are near-zero correlated with each other - they capture independent signals. '
        'Excellent for model performance (low multicollinearity).',
        'TIP'
    )

    # ── Section 12: Age Distributions ──
    pdf.add_page()
    pdf.section_title('12. Age Distributions')
    pdf.add_image_page(f'{GRAPHS_DIR}/12_age_distributions.png')
    pdf.body_text('Account Age (F3887):')
    pdf.bullet_point('Both classes heavily right-skewed - most accounts are **< 6 months old**')
    pdf.bullet_point('Suspicious accounts concentrated in the **0-3 month range** - newly opened')
    pdf.body_text('Customer Age (F3894):')
    pdf.bullet_point('Suspicious accounts peak sharply at **25-35 years**')
    pdf.bullet_point('Legitimate accounts have a broader distribution extending to 70+')

    # ── Section 13: Block Importance ──
    pdf.add_page()
    pdf.section_title('13. Block-Level Importance')
    pdf.add_image_page(f'{GRAPHS_DIR}/13_block_importance.png')
    pdf.body_text('Shows mean and max absolute correlation with target by 500-feature block.')
    pdf.bullet_point('Delta/change blocks (F2500-F2999) show the **highest predictive signal**')

    # ── Section 14: Missing by Class ──
    pdf.section_title('14. Missing Values by Class')
    pdf.add_image_page(f'{GRAPHS_DIR}/14_missing_by_class.png')
    pdf.bullet_point('Legit accounts: mean **1,084** missing features')
    pdf.bullet_point('Suspicious accounts: mean **1,120** missing features - slightly higher')
    pdf.bullet_point('Missingness is **not a strong class signal** but could provide marginal value')

    # ── Summary Table ──
    pdf.add_page()
    pdf.section_title('Summary of Key Visual Insights')
    pdf.ln(3)
    pdf.add_summary_table()
    pdf.ln(10)
    pdf.alert_box(
        'All 14 graphs are saved permanently in files/graphs/ for presentation and documentation purposes.',
        'NOTE'
    )

    # ── Save ──
    pdf.output(OUT_PDF)
    print(f'[DONE] PDF saved to: {OUT_PDF}')


if __name__ == '__main__':
    build_report()
