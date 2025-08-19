# HR Analytics: Employee Satisfaction and Attrition Prediction

## Project Overview
This project addresses two interrelated machine learning tasks in the field of HR analytics for the company **"Care with Work"**:

1. **Regression Task**: Predicting the employee job satisfaction rate (`job_satisfaction_rate`) based on employee characteristics such as department, job level, workload, tenure, career progression, disciplinary records, manager evaluation, and salary.
2. **Classification Task**: Predicting the probability of employee attrition (`quit`) using the same features, enriched by the predicted satisfaction score as an additional predictor.

The objective is to build a predictive system that supports HR decision-making, reduces attrition, and develops personalized retention strategies.

---

## Dataset
- **Size**: 4000 training records, 2000 test records per task  
- **Structure**: 9 features (8 input + 1 target)  
- **Data quality**: Less than 0.2% missing values in categorical features  

---

## Research Workflow
1. **Data loading and preliminary analysis**
2. **Exploratory Data Analysis (EDA)**
   - Job satisfaction strongly correlated with manager evaluation (r = 0.691)  
   - Key attrition risk profile: junior employees, low workload, short tenure (~1.8 years), low salary (-36.6% below average)  
   - Critical factors for attrition: lack of promotions (99.1% of leavers), disciplinary violations (+15.8% to attrition risk)  
3. **Data preprocessing**
   - OrdinalEncoder for hierarchical features (level, workload)  
   - OneHotEncoder for nominal features (dept)  
   - LabelEncoder for binary features (promo, violations)  
   - StandardScaler for numeric features  
4. **Feature engineering**
   - Synthetic feature `job_satisfaction_rate` used in classification  
   - Strong negative correlation with attrition (r = -0.567)  
   - Became the most important predictor (importance = 0.3427)  
5. **Machine Learning Models**
   - Regression: ElasticNet + group medians, SMAPE = 39.17% (target ≤15% not reached due to subjective nature of satisfaction)  
   - Classification: Multiple algorithms tested, feature selection applied, ROC-AUC ≥ 0.91  

---

## Key Findings
### Technical Achievements
- Built a reproducible pipeline for HR analytics with regression and classification models  
- Identified key factors:  
  - Satisfaction: manager evaluation > career growth > absence of violations  
  - Attrition: predicted satisfaction index > tenure > salary  
- Demonstrated interdependence between tasks: satisfaction is a strong predictor of attrition  

### Limitations
- Subjectivity of satisfaction as a human emotion  
- Lack of features on personal motivation, team dynamics, external factors  
- Static models not accounting for temporal labor market trends  

---

## Business Value
### Immediate Value for HR
- **Risk screening**: Identify high-risk employees 3–6 months before attrition  
- **Decision support**: Data-driven prioritization of retention efforts  
- **Economic effect**: Reduced hiring costs, preserved expertise, improved productivity  

### Strategic Recommendations
- **Short-term (1–3 months)**: Improve performance evaluation systems, revise career growth procedures, launch preventive programs for high-risk groups, train managers in feedback and motivation  
- **Medium-term (3–12 months)**: Implement regular pulse surveys, build HR dashboards, design personalized retention programs, integrate external labor market data  
- **Long-term (1+ years)**: Develop a comprehensive People Analytics ecosystem, ML-based behavioral pattern detection, HR analytics competence center, and a data-driven HR culture  

---

## Next Steps
- Pilot deployment of the system in one department with gradual scaling after validating business impact  
- Collect additional data (team dynamics, work-life balance, internal mobility)  
- Experiment with ensemble and deep learning methods for accuracy improvement  
- Enable real-time scoring for new hires  

**Expected ROI**: 150–300% from reduced attrition costs and higher retention of critical employees.  

---

## Installation
```bash
# Clone repository
git clone https://github.com/username/hr-analytics-attrition.git
cd hr-analytics-attrition

# Install dependencies
pip install -r requirements.txt
