# CIBMTR---Equity-in-post-HCT-Survival-Predictions
ML measure to predict the risk score for observations
### This is the Kaggle competition for CIBMTR - Equity in post-HCT Survival Predictions

**Members of this project** :
- Yang Xiang (FrantzXY)
- Rundong Hua (stevenhua0320)
- Siyan Li (Kether0111)
- Enming Yang (EnmingYang)
- Jiacheng He (hej159153)

The link of the competition could be found here:
https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/overview

**High level Goals**
The project is considered to be a success if:
1. The final prediction accuracy is over 85%, if we accomplish to 90%, we got it. 

**Agenda for the meeting**

2024-12-30
- [ ] 1. Register for Kaggle and GH, share this project, brief introduction for GH usage for members that do not know the functionality of GH.
- [x] 2. Enroll in teams in Kaggle
- [x] 3. Background information on the project (life-science, C-index(enhanced one), objective of the model)
- [x] 4. Introduction on the variables on the training set
- [ ] 5. Discussion on the preliminary model that we use to predict. (XGboost in survival analysis, classification work on real event before conducting XGboost?)
- [ ] 6. Schedule in meeting.

2024-12-31
- [ ] 1. Data cleaning, imputation on NA values, what method should we use to impute the categorical values?
- [ ] 2. Since the dimension for the variables is too high, we should do dimension reduction for variables with MCA for categorical variables & LDA for numerical variables.

**Notes for the meeting**


**Reference**
The reference of XGboost: https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html
The reference of C-index: https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/550003
More on the subject of the work: Tushar Deshpande, Deniz Akdemir, Walter Reade, Ashley Chow, Maggie Demkin, and Yung-Tsi Bolon. CIBMTR - Equity in post-HCT Survival Predictions. https://kaggle.com/competitions/equity-post-HCT-survival-predictions, 2024. Kaggle.
