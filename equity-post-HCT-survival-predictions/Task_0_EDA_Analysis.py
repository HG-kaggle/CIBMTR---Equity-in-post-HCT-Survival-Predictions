# %%

# Author Yang Xiang, Date 2024 12 30.
# All the group members have been added to the project
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mp
import seaborn as sns
import matplotlib.pyplot as plt  # 
import scipy as sp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os


## Part 1: Data cleaning, adding -1 for numerical missing values, and "NA" string value
## for categorical missing values.
train = pd.read_csv('equity-post-HCT-survival-predictions/train.csv')
# This is for test purpose to fix

# %%

# 定义需要转换的列
columns_to_encode = [
    'psych_disturb', 'diabetes', 'arrhythmia', 'renal_issue', 'pulm_severe', 
    'rituximab', 'obesity', 'in_vivo_tcd', 'hepatic_severe', 'prior_tumor', 
    'peptic_ulcer', 'rheum_issue', 'hepatic_mild', 'cardiac', 'pulm_moderate'
]

# 将 Yes/No 转换为 1/0
for col in columns_to_encode:
    train[col] = train[col].replace({'Yes': 1, 'No': 0})

# %%

columns_to_encode = [
    'dri_score', 'cyto_score', 'tbi_status', 'graft_type', 'prim_disease_hct', 
    'cmv_status', 'tce_imm_match', 'prod_type', 'cyto_score_detail', 
    'conditioning_intensity', 'mrd_hct', 'tce_match', 'gvhd_proph', 
    'sex_match', 'race_group', 'tce_div_match', 'melphalan_dose','vent_hist', 'ethnicity'
]

# 初始化 OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

# 对指定列进行独热编码
encoded_data = encoder.fit_transform(train[columns_to_encode])

# 获取编码后的列名
encoded_column_names = encoder.get_feature_names_out(columns_to_encode)

# 将编码后的数据转换为 DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_names)

# 将编码后的数据合并回原始 DataFrame
data = pd.concat([train.drop(columns=columns_to_encode), encoded_df], axis=1)

# 查看生成的新变量
print("Encoded Column Names:")
print(encoded_column_names)


# %%
print(data.columns)
with open("column_names.txt", "w") as file:
    for col in data.columns:
        file.write(col + "\n")

print("Column names have been saved to 'column_names.txt'.")

# %%

group_columns = [
    'ethnicity_Non-resident of the U.S.',
    'ethnicity_Not Hispanic or Latino', 'ethnicity_nan', 'age_at_hct', 'donor_age', 'donor_related',
    'race_group_White', 'race_group_Asian', 'race_group_Black or African-American',
    'sex_match_M-F', 'sex_match_F-M', 'sex_match_M-M'
]

# 删除包含 'Not done' 的行
data = data[~data.isin(['Not done']).any(axis=1)]

# 检查分组变量中是否有缺失值
data['gen'] = data[group_columns].isnull().any(axis=1).astype(int)

# 过滤数据：只保留没有缺失值的行
filtered_data = data[data['gen'] == 0].drop(columns=['gen'])

# 确保所有列都是数值型
filtered_data = filtered_data[group_columns].select_dtypes(include=['number'])

# 计算相关性矩阵
correlation_matrix = filtered_data.corr()

# 绘制热力图
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
plt.title("Correlation Heatmap for Grouped Variables (No Missing Values)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%


group_columns = [
    'comorbidity_score', 'karnofsky_score', 'diabetes', 'obesity', 'prior_tumor',
    'hepatic_severe', 'hepatic_mild', 'renal_issue', 'pulm_severe', 'pulm_moderate',
    'cardiac', 'arrhythmia', 'psych_disturb',  'vent_hist_Yes', 'vent_hist_nan' , 'peptic_ulcer', 'rheum_issue',
    'dri_score_High - TED AML case <missing cytogenetics',
    'dri_score_Intermediate',
    'dri_score_Intermediate - TED AML case <missing cytogenetics',
    'dri_score_Low',
    'dri_score_Missing disease status',
    'dri_score_N/A - disease not classifiable',
    'dri_score_N/A - non-malignant indication',
    'dri_score_N/A - pediatric',
    'dri_score_TBD cytogenetics',
    'dri_score_Very high',
    'dri_score_nan'
]

# 删除包含 'Not done' 的行
data = data[~data.isin(['Not done']).any(axis=1)]

# 检查分组变量中是否有缺失值
data['gen'] = data[group_columns].isnull().any(axis=1).astype(int)

# 过滤数据：只保留没有缺失值的行
filtered_data = data[data['gen'] == 0].drop(columns=['gen'])

# 确保所有列都是数值型
filtered_data = filtered_data[group_columns].select_dtypes(include=['number'])

# 检查是否所有列都成功转换为数值型
if len(filtered_data.columns) != len(group_columns):
    missing_columns = [col for col in group_columns if col not in filtered_data.columns]
    print(f"Warning: Some columns were not included due to non-numeric data: {missing_columns}")

# 计算相关性矩阵
correlation_matrix = filtered_data.corr()

# 绘制热力图
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
plt.title("Correlation Heatmap for Grouped Variables (No Missing Values)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
group_columns = [
     'prim_disease_hct_ALL',
    'prim_disease_hct_AML',
    'prim_disease_hct_CML',
    'prim_disease_hct_HD',
    'prim_disease_hct_HIS',
    'prim_disease_hct_IEA',
    'prim_disease_hct_IIS',
    'prim_disease_hct_IMD',
    'prim_disease_hct_IPA',
    'prim_disease_hct_MDS',
    'prim_disease_hct_MPN',
    'prim_disease_hct_NHL',
    'prim_disease_hct_Other acute leukemia',
    'prim_disease_hct_Other leukemia',
    'prim_disease_hct_PCD',
    'prim_disease_hct_SAA',
    'prim_disease_hct_Solid tumor',
    'year_hct',
    'graft_type_Peripheral blood',
    'conditioning_intensity_N/A, F(pre-TED) not submitted',
    'conditioning_intensity_NMA',
    'conditioning_intensity_No drugs reported',
    'conditioning_intensity_RIC',
    'conditioning_intensity_TBD',
    'conditioning_intensity_nan',
    'in_vivo_tcd',
     'gvhd_proph_CDselect alone',
    'gvhd_proph_CSA + MMF +- others(not FK)',
    'gvhd_proph_CSA + MTX +- others(not MMF,FK)',
    'gvhd_proph_CSA +- others(not FK,MMF,MTX)',
    'gvhd_proph_CSA alone',
    'gvhd_proph_Cyclophosphamide +- others',
    'gvhd_proph_Cyclophosphamide alone',
    'gvhd_proph_FK+ MMF +- others',
    'gvhd_proph_FK+ MTX +- others(not MMF)',
    'gvhd_proph_FK+- others(not MMF,MTX)',
    'gvhd_proph_FKalone',
    'gvhd_proph_No GvHD Prophylaxis',
    'gvhd_proph_Other GVHD Prophylaxis',
    'gvhd_proph_Parent Q = yes, but no agent',
    'gvhd_proph_TDEPLETION +- other',
    'gvhd_proph_TDEPLETION alone',
    'gvhd_proph_nan',
    'mrd_hct_Positive',
    'mrd_hct_nan',
    'cmv_status_+/-',
    'cmv_status_-/+',
    'cmv_status_-/-',
    'cmv_status_nan',
]

# 删除包含 'Not done' 的行
data = data[~data.isin(['Not done']).any(axis=1)]

# 检查分组变量中是否有缺失值
data['gen'] = data[group_columns].isnull().any(axis=1).astype(int)

# 过滤数据：只保留没有缺失值的行
filtered_data = data[data['gen'] == 0].drop(columns=['gen'])

# 确保所有列都是数值型
filtered_data = filtered_data[group_columns].select_dtypes(include=['number'])

# 检查是否所有列都成功转换为数值型
if len(filtered_data.columns) != len(group_columns):
    missing_columns = [col for col in group_columns if col not in filtered_data.columns]
    print(f"Warning: Some columns were not included due to non-numeric data: {missing_columns}")

# 计算相关性矩阵
correlation_matrix = filtered_data.corr()

# 绘制热力图
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
plt.title("Correlation Heatmap for Grouped Variables (No Missing Values)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
group_columns = [
     'hla_match_a_high',
    'hla_match_b_high',
    'hla_match_c_high',
    'hla_match_a_low',
    'hla_match_b_low',
    'hla_match_c_low',
    'hla_match_drb1_high',
    'hla_match_drb1_low',
    'hla_match_dqb1_high',
    'hla_match_dqb1_low',
    'hla_high_res_6',
    'hla_high_res_8',
    'hla_high_res_10',
    'hla_low_res_6',
    'hla_low_res_8',
    'hla_low_res_10',
    'hla_nmdp_6',
    'tce_imm_match_G/G',
    'tce_imm_match_H/B',
    'tce_imm_match_H/H',
    'tce_imm_match_P/B',
    'tce_imm_match_P/G',
    'tce_imm_match_P/H',
    'tce_imm_match_P/P',
    'tce_imm_match_nan',
    'tce_div_match_GvH non-permissive',
    'tce_div_match_HvG non-permissive',
    'tce_div_match_Permissive mismatched',
    'tce_div_match_nan',
    'tce_match_GvH non-permissive',
    'tce_match_HvG non-permissive',
    'tce_match_Permissive',
    'tce_match_nan',
]

# 删除包含 'Not done' 的行
data = data[~data.isin(['Not done']).any(axis=1)]

# 检查分组变量中是否有缺失值
data['gen'] = data[group_columns].isnull().any(axis=1).astype(int)

# 过滤数据：只保留没有缺失值的行
filtered_data = data[data['gen'] == 0].drop(columns=['gen'])

# 确保所有列都是数值型
filtered_data = filtered_data[group_columns].select_dtypes(include=['number'])

# 检查是否所有列都成功转换为数值型
if len(filtered_data.columns) != len(group_columns):
    missing_columns = [col for col in group_columns if col not in filtered_data.columns]
    print(f"Warning: Some columns were not included due to non-numeric data: {missing_columns}")

# 计算相关性矩阵
correlation_matrix = filtered_data.corr()

# 绘制热力图
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
plt.title("Correlation Heatmap for Grouped Variables (No Missing Values)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%

group_columns = [
    'rituximab',
    'melphalan_dose_N/A, Mel not given',
    'melphalan_dose_nan',
    'prod_type_PB',
     'tbi_status_TBI + Cy +- Other',
    'tbi_status_TBI +- Other, -cGy, fractionated',
    'tbi_status_TBI +- Other, -cGy, single',
    'tbi_status_TBI +- Other, -cGy, unknown dose',
    'tbi_status_TBI +- Other, <=cGy',
    'tbi_status_TBI +- Other, >cGy',
    'tbi_status_TBI +- Other, unknown dose',
    'cyto_score_Intermediate',
    'cyto_score_Normal',
    'cyto_score_Not tested',
    'cyto_score_Other',
    'cyto_score_Poor',
    'cyto_score_TBD',
    'cyto_score_nan',
    'cyto_score_detail_Intermediate',
'cyto_score_detail_Not tested',
'cyto_score_detail_Poor',
'cyto_score_detail_TBD',
'cyto_score_detail_nan',

]

# 删除包含 'Not done' 的行
data = data[~data.isin(['Not done']).any(axis=1)]

# 检查分组变量中是否有缺失值
data['gen'] = data[group_columns].isnull().any(axis=1).astype(int)

# 过滤数据：只保留没有缺失值的行
filtered_data = data[data['gen'] == 0].drop(columns=['gen'])

# 确保所有列都是数值型
filtered_data = filtered_data[group_columns].select_dtypes(include=['number'])

# 检查是否所有列都成功转换为数值型
if len(filtered_data.columns) != len(group_columns):
    missing_columns = [col for col in group_columns if col not in filtered_data.columns]
    print(f"Warning: Some columns were not included due to non-numeric data: {missing_columns}")

# 计算相关性矩阵
correlation_matrix = filtered_data.corr()

# 绘制热力图
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
plt.title("Correlation Heatmap for Grouped Variables (No Missing Values)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


