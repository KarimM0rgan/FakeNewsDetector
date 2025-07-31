# Fake News Detection Program

## Overview
A machine learning system that detects fake news articles with 99.2% accuracy, including detailed error analysis of:
- **False Positives**: Legitimate news incorrectly flagged as fake
- **False Negatives**: Fake news classified as legit

## Key Features
- **Text Preprocessing Pipeline**: Cleaning, normalization, and feature extraction
- **Dual Error Tracking**: 
  - `false_positives`: Real news misclassified as fake
  - `false_negatives`: Fake news the model missed
  - `misclassified_news.csv`: All misclassified news combined and exported into a .csv file for further analysis
- **Model Interpretability**: Includes word frequency analysis for errors
- **Deployment Ready**: Flask/Streamlit compatible

## Error Analysis Results
```python
Confusion Matrix:
[[4687   20]  ← 4687 correct (REAL), 20 false (FAKE)
 [  23 4250]] ← 4250 Fake news correctly identified, 23 Real

Accuracy: 99.22%
False Positives: 20 (0.43% of real news)
False Negatives: 23 (0.54% of fake news)
```
---

## Error Types Deep Dive

### False Positives (Type I Errors)
**Definition**: Real news incorrectly flagged as fake  
**Example**:
```text
"NASA's Perseverance rover finds organic molecules on Mars" 
→ Labeled fake by model
```
**Mitigation Strategies**:
- Add trusted source whitelist

### False Negatives (Type II Errors)  
**Definition**: Fake news mistakenly classified as real  
**Example**: 
```text
"Vaccines contain microchips, says Pfizer insider"
→ Labeled real by model  
```
**Mitigation Strategies**:
- Add fact-checking API integration

---
## Visualization
![Pie Chart](E:\Sewanee\Data Analysis\.py\Fake News Detector\FakeNewsDetector\Misclassified News Chart.png)

---

### 🔍 Why This Project Matters

**1. Combating Misinformation**  
With 67% of Americans encountering fake news regularly (*Pew Research*), this tool helps:
- Identify viral misinformation patterns
- Surface commonly faked topics through error analysis
- Provide transparency in classification decisions

**2. Technical Skills Showcase**  
- **NLP Pipeline**: TF-IDF, text preprocessing, model interpretability
- **Error Analysis**: False positive/negative diagnostics
- **Ethic Use of Technology**

**3. Real-World Impact**  
The dual error tracking enables:
- News platforms to reduce over-censorship (false positives)
- Fact-checkers to prioritize review of likely misses (false negatives)
- Researchers to study misinformation trends
---
