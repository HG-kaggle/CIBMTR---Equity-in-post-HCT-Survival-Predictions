# CIBMTR---Equity-in-post-HCT-Survival-Predictions
ML measure to predict the risk score for observations
### This is the Kaggle competition for CIBMTR - Equity in post-HCT Survival Predictions

**Members of this project** :
- Yang Xiang (FrantzXY)
- Rundong Hua (stevenhua0320)
- Siyan Li (Kether0111)
- Enming Yang (EnmingYang)
- Jiacheng He (hej159753)

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

2025-01-01
- [ ] 1. clustering algorithm analysis for complex events. (efs is a complex event, incorporating death, replapse ...对年龄种族和efs是否等于零做好分类后再依据变量区分风险)

2025-01-02
- [x] 1. Understand the process of HCT, understand the variables in depth. (Rundong Hua, Siyan Li)

### Notes for the meeting
**1. Data cleaning & Imputation**

对于数值型特征使用平均值来填充，类别型特征使用众数来填充（刘楷林和尚培培，2021）

KNN填充
利用knn算法填充，其实是把目标列当做目标标量，利用非缺失的数据进行knn算法拟合，最后对目标列缺失进行预测。（对于连续特征一般是加权平均，对于离散特征一般是加权投票）

随机森林填充
随机森林算法填充的思想和knn填充是类似的，即利用已有数据拟合模型，对缺失变量进行预测。
数据不能直接全部都填补，如果这个空缺是事出有因的，那么就要留意一下是否空缺本身包含信息
需要从原理上思考一下为什么空缺
数据挖掘算法本身更致力于避免数据过分适合所建的模型，这一特性使得它难以通过自身的算法去很好地处理不完整数据。因此，空缺的数据需要通过专门的方法进行推导、填充等，以减少数据挖掘算法与实际应用之间的差距。

K最近距离邻法（K-means clustering）
先根据欧式距离或相关分析来确定距离具有缺失数据样本最近的K个样本，将这K个值加权平均来估计该样本的缺失数据。
同均值插补的方法都属于单值插补，不同的是，它用层次聚类模型预测缺失变量的类型，再以该类型的均值插补。假设X=(X1,X2…Xp)为信息完全的变量，Y为存在缺失值的变量，那么首先对X或其子集行聚类，然后按缺失个案所属类来插补不同类的均值。如果在以后统计分析中还需以引入的解释变量和Y做分析，那么这种插补方法将在模型中引入自相关，给分析造成障碍。

直接在包含空值的数据上进行数据挖掘。这类方法包括贝叶斯网络和人工神经网络等。
贝叶斯网络是用来表示变量间连接概率的图形模式，它提供了一种自然的表示因果信息的方法，用来发现数据间的潜在关系。在这个网络中，用节点表示变量，有向边表示变量间的依赖关系。贝叶斯网络仅适合于对领域知识具有一定了解的情况，至少对变量间的依赖关系较清楚的情况。否则直接从数据中学习贝叶斯网的结构不但复杂性较高（随着变量的增加，指数级增加），网络维护代价昂贵，而且它的估计参数较多，为系统带来了高方差，影响了它的预测精度。
当在任何一个对象中的缺失值数量很大时，存在指数爆炸的危险。（在我们的case下有梯度/指数爆炸的风险）

** Variable Explain **
Mostly could be found in excel file and screenshot of notes in Wechat group (2025-01-02)
Here we explain what in notes not covered.

graft type(强调干细胞移植位置策略)：Peripheral blood (PB), Bone Marrow (BM)
product type(强调干细胞供体位置来源)： Peripheral blood (PB), Bone Marrow (BM)
Notice that even though their categories are the same, they carry different meanings

Ex: 患者A可以从捐赠者B获得Bone Marrow位置的造血干细胞，但是因为身体原因，只能从peripheral blood处
进行造血干细胞移植，这里对于患者A进行分析，它的graft type是PB，但是product type是BM.




**Reference**
The reference of XGboost: https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html
The reference of C-index: https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/550003
More on the subject of the work: Tushar Deshpande, Deniz Akdemir, Walter Reade, Ashley Chow, Maggie Demkin, and Yung-Tsi Bolon. CIBMTR - Equity in post-HCT Survival Predictions. https://kaggle.com/competitions/equity-post-HCT-survival-predictions, 2024. Kaggle.
