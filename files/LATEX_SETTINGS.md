# LaTeX Document Settings — Professional Proposal

## Recommended Settings for MuleShield AI Proposal

---

## 1. Document Class & Page Setup

```latex
\documentclass[11pt, a4paper]{article}
```

| Setting | Value | Reason |
|---------|-------|--------|
| **Font size** | `11pt` | Industry standard for proposals. 12pt is for thesis/dissertation. 10pt is too small for reading comfort |
| **Paper** | `a4paper` | Indian standard. Use `letterpaper` only for US submissions |
| **Class** | `article` | Best for 10-30 page proposals. Use `report` for 50+ pages |

---

## 2. Margins

```latex
\usepackage[
    top=1in,
    bottom=1in,
    left=1.25in,
    right=1in
]{geometry}
```

| Setting | Value | Reason |
|---------|-------|--------|
| **Top/Bottom** | 1 inch (2.54 cm) | Standard across IEEE, ACM, IIT templates |
| **Left** | 1.25 inch | Slightly wider for binding margin (if printed) |
| **Right** | 1 inch | Standard |
| **Alternative** | `margin=1in` (all sides equal) | If no binding needed |

---

## 3. Font Family (THE Most Important Choice)

### Option A: Palatino (Top Recommendation ⭐)
```latex
\usepackage{newpxtext}   % Palatino-based text
\usepackage{newpxmath}   % Matching math
```
- **Why**: Hermann Zapf's Palatino is THE premium academic font
- **Feel**: Elegant, substantial, professional — used by Cambridge University Press
- **Readability**: Slightly wider characters than Times → better readability

### Option B: Libertinus (Modern Alternative)
```latex
\usepackage{libertinus}
```
- **Why**: Modern, beautiful, excellent math support
- **Feel**: Clean, contemporary — popular in CS/AI papers
- **Readability**: Outstanding; designed for technical documents

### Option C: STIX Two (Scientific/Technical)
```latex
\usepackage{stix2}
```
- **Why**: Designed specifically for scientific publishing by STI Pub
- **Feel**: Similar to Times but more refined
- **Readability**: Optimized for mixed text + math

### ❌ Fonts to AVOID
- **Computer Modern** (LaTeX default) — too thin, looks "grad student homework"
- **Times New Roman** — overused, looks like MS Word
- **Arial/Helvetica for body** — sans-serif is harder to read in long documents
- **Comic Sans, Courier** — never

---

## 4. Typography Micro-Optimization

```latex
\usepackage[T1]{fontenc}          % Better font encoding
\usepackage[utf8]{inputenc}       % Unicode support
\usepackage{microtype}            % CRITICAL: micro-typography
\usepackage[english]{babel}       % Proper hyphenation
```

| Package | Effect | Impact |
|---------|--------|--------|
| **microtype** | Character protrusion, font expansion | Lines look 30% cleaner; eliminates awkward word spacing. THE single most impactful package |
| **fontenc T1** | Proper glyph encoding | Correct hyphenation, accented characters |

---

## 5. Paragraph & Line Spacing

```latex
\usepackage{setspace}
\onehalfspacing                   % 1.5x line spacing

\usepackage{parskip}              % Space between paragraphs instead of indent
% OR manually:
% \setlength{\parindent}{0pt}
% \setlength{\parskip}{6pt plus 2pt minus 1pt}
```

| Setting | Value | Reason |
|---------|-------|--------|
| **Line spacing** | 1.5x (`\onehalfspacing`) | Sweet spot between dense and airy. Single=too dense; Double=too wasteful |
| **Paragraph style** | Block paragraphs (no indent, space between) | Modern, professional look. Traditional indent looks academic |
| **Paragraph skip** | 6pt | Enough separation without looking scattered |

---

## 6. Color Scheme (Professional & Restrained)

```latex
\usepackage[table]{xcolor}

% Primary palette — Professional Blue/Grey
\definecolor{mainblue}{RGB}{0, 70, 127}       % Deep corporate blue
\definecolor{darkgrey}{RGB}{51, 51, 51}        % Near-black for body text
\definecolor{accentblue}{RGB}{41, 128, 185}    % Lighter accent blue
\definecolor{highlightgreen}{RGB}{39, 174, 96} % Success/positive indicators
\definecolor{warningred}{RGB}{192, 57, 43}     % Alert/critical indicators
\definecolor{lightgrey}{RGB}{245, 245, 245}    % Table row shading
\definecolor{tablebg}{RGB}{230, 240, 250}      % Light blue table headers
```

### Rules:
- **Body text**: Dark grey (`darkgrey`) — not pure black (less eye strain)
- **Section headings**: `mainblue` — gives visual hierarchy
- **Tables**: Alternate `lightgrey` rows (`\rowcolors{2}{white}{lightgrey}`)
- **Links**: `accentblue` — clickable but not distracting
- **Never**: Use more than 3 colors on a single page

---

## 7. Section Headings

```latex
\usepackage{titlesec}

% Section: Large, bold, blue, with rule below
\titleformat{\section}
  {\Large\bfseries\color{mainblue}}  % format
  {\thesection.}                      % label
  {0.5em}                             % separator
  {}                                   % before-code
  [\vspace{2pt}\titlerule]            % after-code (underline)

% Subsection: medium, bold, dark grey
\titleformat{\subsection}
  {\large\bfseries\color{darkgrey}}
  {\thesubsection}
  {0.5em}
  {}

% Spacing: {left}{before}{after}
\titlespacing*{\section}{0pt}{20pt}{10pt}
\titlespacing*{\subsection}{0pt}{14pt}{6pt}
```

---

## 8. Headers & Footers

```latex
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}                                        % Clear defaults
\fancyhead[L]{\small\color{darkgrey} MuleShield AI — Idea Proposal}
\fancyhead[R]{\small\color{darkgrey} Team SK8-infi}
\fancyfoot[C]{\small\color{darkgrey} \thepage}
\renewcommand{\headrulewidth}{0.4pt}               % Thin header line
\renewcommand{\footrulewidth}{0pt}                  % No footer line
```

---

## 9. Tables (Professional)

```latex
\usepackage{booktabs}     % \toprule, \midrule, \bottomrule
\usepackage{tabularx}     % Full-width tables
\usepackage{multirow}     % Multi-row cells

\renewcommand{\arraystretch}{1.3}  % Taller rows (1.0 = default, too cramped)
```

### Rules for Professional Tables:
- ✅ Use `booktabs` rules (horizontal only: `\toprule`, `\midrule`, `\bottomrule`)
- ✅ Alternate row shading: `\rowcolors{2}{white}{lightgrey}`
- ✅ Header row in bold or colored background
- ❌ Never use vertical lines (`|`)
- ❌ Never use `\hline` (use `booktabs` instead)

---

## 10. Figures & Diagrams

```latex
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{positioning, shapes.geometric, arrows.meta, calc}

\usepackage{float}        % [H] placement
\usepackage{caption}      % Better captions
\captionsetup{
    font=small,
    labelfont=bf,
    skip=8pt
}
```

---

## 11. Lists (Compact & Clean)

```latex
\usepackage{enumitem}
\setlist{
    topsep=4pt,
    itemsep=2pt,
    parsep=0pt,
    leftmargin=20pt
}
% Custom icons for itemize:
\setlist[itemize,1]{label=\textcolor{mainblue}{\textbullet}}
```

---

## 12. Links & References

```latex
\usepackage[
    colorlinks=true,
    linkcolor=mainblue,
    urlcolor=accentblue,
    citecolor=mainblue
]{hyperref}
```

---

## 13. Code Blocks (If Needed)

```latex
\usepackage{listings}
\lstset{
    backgroundcolor=\color{lightgrey},
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    rulecolor=\color{darkgrey!30},
    numbers=left,
    numberstyle=\tiny\color{darkgrey!50},
    keywordstyle=\color{mainblue}\bfseries,
    commentstyle=\color{highlightgreen},
    stringstyle=\color{warningred}
}
```

---

## 14. Complete Preamble (Copy-Paste Ready)

```latex
\documentclass[11pt, a4paper]{article}

% === GEOMETRY ===
\usepackage[top=1in, bottom=1in, left=1.25in, right=1in]{geometry}

% === FONTS & TYPOGRAPHY ===
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newpxtext}              % Palatino text
\usepackage{newpxmath}              % Palatino math
\usepackage{microtype}              % Micro-typography (ESSENTIAL)
\usepackage[english]{babel}

% === SPACING ===
\usepackage{setspace}
\onehalfspacing
\usepackage{parskip}

% === COLORS ===
\usepackage[table]{xcolor}
\definecolor{mainblue}{RGB}{0, 70, 127}
\definecolor{darkgrey}{RGB}{51, 51, 51}
\definecolor{accentblue}{RGB}{41, 128, 185}
\definecolor{highlightgreen}{RGB}{39, 174, 96}
\definecolor{warningred}{RGB}{192, 57, 43}
\definecolor{lightgrey}{RGB}{245, 245, 245}
\definecolor{tablebg}{RGB}{230, 240, 250}

% === HEADINGS ===
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\color{mainblue}}{\thesection.}{0.5em}{}[\vspace{2pt}\titlerule]
\titleformat{\subsection}{\large\bfseries\color{darkgrey}}{\thesubsection}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{darkgrey}}{\thesubsubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{20pt}{10pt}
\titlespacing*{\subsection}{0pt}{14pt}{6pt}

% === HEADER/FOOTER ===
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{darkgrey} MuleShield AI — Idea Proposal}
\fancyhead[R]{\small\color{darkgrey} Team SK8-infi}
\fancyfoot[C]{\small\color{darkgrey} \thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% === TABLES ===
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\renewcommand{\arraystretch}{1.3}

% === FIGURES & DIAGRAMS ===
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{positioning, shapes.geometric, arrows.meta, calc, fit, backgrounds}
\usepackage{float}
\usepackage{caption}
\captionsetup{font=small, labelfont=bf, skip=8pt}

% === LISTS ===
\usepackage{enumitem}
\setlist{topsep=4pt, itemsep=2pt, parsep=0pt, leftmargin=20pt}
\setlist[itemize,1]{label=\textcolor{mainblue}{\textbullet}}

% === LINKS ===
\usepackage[colorlinks=true, linkcolor=mainblue, urlcolor=accentblue, citecolor=mainblue]{hyperref}

% === CODE ===
\usepackage{listings}
\lstset{
    backgroundcolor=\color{lightgrey},
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    rulecolor=\color{darkgrey!30}
}

% === METADATA ===
\title{\textbf{\textcolor{mainblue}{MuleShield AI}}\\[6pt]
\large AI/ML-Based Classification of Suspicious Mule Accounts\\[4pt]
\normalsize PSB Cybersecurity, Fraud \& AI Hackathon 2026 — Problem Statement 2}
\author{Team SK8-infi}
\date{June 2026}
```

---

## Quick Reference Summary

| Setting | Optimal Value | Why |
|---------|--------------|-----|
| **Font size** | 11pt | Professional standard for proposals |
| **Font family** | Palatino (newpxtext) | Premium, readable, substantial |
| **Margins** | 1" top/bottom/right, 1.25" left | Clean with binding margin |
| **Line spacing** | 1.5x (onehalfspacing) | Readable without wasting space |
| **Paragraph style** | Block (parskip, no indent) | Modern professional look |
| **Colors** | Max 3: blue headings + dark grey text + light grey tables | Restrained = premium |
| **Tables** | booktabs (horizontal rules only, no vertical lines) | Academic gold standard |
| **Micro-typography** | microtype package | Single most impactful visual upgrade |
| **Headings** | Blue, bold, with underline rule | Clear visual hierarchy |
| **Header/footer** | Document title left, team right, page center | Professional navigation |
