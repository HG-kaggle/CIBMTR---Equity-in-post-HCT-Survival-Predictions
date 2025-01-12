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

2025-01-12
This week's action:
- [ ] 1. Tackling the ambiguous variables. variable整理
- [ ] 2. **推翻之前所有clustering，catboost工作，没有意义。**
- [ ] 3. 先从理解项目目标开始。见“https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/550003” 帖子。C-Index不需要去考虑efs=0时候efs_time的不确定性。我们只需要确定efs_time严格意义上小于或者大于另外一个efs_time的情况。比如，见上链接帖子，C-Index Denominator里面没有D-F这一项，因为F是efs=0,虽然F的efs_time比D的短，但是我们不知道F是在D前面还是后面发病了，我们只知道F在efs_time之前没有发病。**所以，我们之前的误区就是我们尝试去quantify efs=0的不确定性。但是显然，主办方的C-Index显示我们不需要去比较不确定性的efs=0的time，我们只需要确定严格意义上小于或者大于另外一个efs_time的情况。** 所以，我们需要做的是classify efs=1 and efs=0, 然后根据classification的efs=1/0, 将efs=1 和efs=0的两个情况分开来做regression predict他们的efs_time.
- [ ] 4. 现在主要矛盾和问题是如何解决缺失值。（classification & regression的数据不能有NA）

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

BIRCH算法
BIRCH算法(平衡迭代削减聚类法)：聚类特征使用3元组进行一个簇的相关信息，通过构建满足分枝因子和簇直径限制的聚类特征树来求聚类，聚类特征树其实是一个具有两个参数分枝因子和类直径的高度平衡树；分枝因子规定了树的每个节点的子女的最多个数，而类直径体现了对这一类点的距离范围；非叶子节点为它子女的最大特征值；聚类特征树的构建可以是动态过程的，可以随时根据数据对模型进行更新操作。
优缺点：
适合大规模数据集，线性效率；
只适合分布呈凸形或者球形的数据集、需要给定聚类个数和簇之间的相关参数；
2. Gower 距离的定义非常简单。首先每个类型的变量都有特殊的距离度量方法，而且该方法会将变量标准化到[0,1]之间。接下来，利用加权线性组合的方法来计算最终的距离矩阵。不同类型变量的计算方法如下所示：
连续型变量：利用归一化的曼哈顿距离
顺序型变量：首先将变量按顺序排列，然后利用经过特殊调整的曼哈顿距离
名义型变量：首先将包含 k 个类别的变量转换成 k 个 0-1 变量，然后利用 Dice 系数做进一步的计算优点：通俗易懂且计算方便
缺点：非常容易受无标准化的连续型变量异常值影响，所以数据转换过程必不可少；该方法需要耗费较大的内存
k-modes的优点： 可适用于离散性数据集、时间复杂度更低。k-modes的缺点： 需要事先对k值进行确定。
3. 算法优化 ：agglomerate algorithm的算法优化。


** Variable Explain **
Mostly could be found in excel file and screenshot of notes in Wechat group (2025-01-02)
Here we explain what in notes not covered.

graft type(强调干细胞移植位置策略)：Peripheral blood (PB), Bone Marrow (BM)
product type(强调干细胞供体位置来源)： Peripheral blood (PB), Bone Marrow (BM)
Notice that even though their categories are the same, they carry different meanings

Ex: 患者A可以从捐赠者B获得Bone Marrow位置的造血干细胞，但是因为身体原因，只能从peripheral blood处
进行造血干细胞移植，这里对于患者A进行分析，它的graft type是PB，但是product type是BM.


**Problems in data cleaning**

1. Some levels are ambiguous, variables are: conditioning density, prim_disease_hct, tbi_status, gvhd_proph

**Reference**
The reference of XGboost: https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html
The reference of Catboost: https://github.com/catboost/catboost/blob/master/catboost/tutorials/categorical_features/categorical_features_parameters.ipynb
The reference of C-index: https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/550003
More on the subject of the work: Tushar Deshpande, Deniz Akdemir, Walter Reade, Ashley Chow, Maggie Demkin, and Yung-Tsi Bolon. CIBMTR - Equity in post-HCT Survival Predictions. https://kaggle.com/competitions/equity-post-HCT-survival-predictions, 2024. Kaggle.
