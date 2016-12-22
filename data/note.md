- 有46个分类变量，包括23个名词和23个序数变量，在数据集中有33个数字变量。
- neighborhood, zoning, house quality and facility might distinguish the house value
- 一些区域相关的功能，如批量面积，一楼平方英尺，二楼平方英尺和房屋建成显示与销售价格正相关
- 我们将建模分为两个部分。 一方面，我们建模以实现高预测准确性，另一方面，我们建模以保持解释。 我们首先讨论聚焦于实现高预测准确性的建模。 作为第一步，我们调整了所有基础学习者的参数。

# 特征选择
- GrLivArea: 生活区面积 int64，不缺
- TotalBsmtSF：地下室面积，int64，不缺
- OverallQual：评估房子的整体材料和表面质量，int不缺
- 1stFlrSF：1楼面积，int不缺
- LotArea：地段面积，int64不缺
- 2ndFlrSF：二楼面积，int64不缺
- BsmtFinSF1  ：地下室成品面积（type1)int64不缺
- GarageArea ：车库面积 int64 ,不缺
- GarageCars：车库车容量，int64，不缺
- YearBuilt : 建造年份，int64， 不缺
- TotRmsAbvGrd：所有房间数目，不含bathroom，int64，不缺
- MSSubClass: 住宅类型,和新旧就有关，int64，不缺，考虑是否向量化
- Fireplaces：壁炉数目，int64，不缺
- FullBath:卫生间数目，int64,不缺
- FireplaceQu：壁炉质量，object，缺少，置NA，向量化
       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
- ExterQual：评估外部材料的质量 object，不缺，向量化
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
- Neighborhood object，不缺，向量化
- YearRemodAdd：重建日期（如果没有重建或增加，与施工日期相同）int64，不缺
- GarageYrBlt：车库创建日期，float64，缺，众数
- OverallCond: 评价房子的整体情况 int64，不缺
- KitchenQual: Kitchen quality，object，不缺，向量化
- MSZoning：标识销售的一般分区分类，object，不缺，向量化
- BedroomAbvGr：地上卧室数目 int64，不缺
- GarageType：车库类型，缺少，置NA，向量化。
- BldgType：object，不缺，向量化
- HalfBath：int64,不缺
- MasVnrArea：砌体胶合板面积平方英尺，int64，缺少，置0
- BsmtQual：地下室质量：缺少，置NA，向量化
- GarageFinish：车库内部装饰，缺少，置NA，向量化
- OpenPorchSF：int64，不缺
- HouseStyle：不缺，向量化
- BsmtUnfSF：int64，不缺
- BsmtFinType1：缺少，置NA，向量化
- LotFrontage：float64，缺少，取均值
- Foundation：object不缺，向量化
- HeatingQC：object不缺，向量化
- WoodDeckSF：int64不缺，
- CentralAir：object，转换为0,1，不缺
- GarageQual：object，缺少，置NA，向量化
- BsmtCond：object，缺少，置NA，向量化

- BsmtExposure：object，地下室曝光，缺少，置NA，向量化
- Exterior1st：房顶材质，object，不缺，向量化
- KitchenAbvGr：int64，不缺
- Exterior2nd：房顶材质，object，不缺，向量化
- LotShape：房屋形状，object，不缺，向量化
- RoofStyle:房顶样式，object，不缺，向量化
- MasVnrType：砌体单板类型，object，缺少，置众数，向量化
- BsmtFinType2：object，缺少，置NA，向量化
- PavedDrive：object，不缺，向量化
- ScreenPorch：int64，不缺
- Functional：object，不缺，向量化
- LandContour：object，不缺，向量化
- Condition1：object，不缺，向量化
- Fence：object，大量缺少，置NA，向量化。
- SaleCondition：object，不缺，向量化
- Electrical：object，缺少，置众数
- LandSlope：object，不缺，向量化

# 统计计算出的额外属性
- RemodAge：from YearRemodAdd，不缺
- houseAge：from YearBuilt，不缺
- GarageAge：from GarageYrBlt，缺少，取均值或最大值（先不選这个特征）

# 随机森林超参结果
- 网上推荐的结果：{'max_depth': 15, 'n_estimators': 150, 'max_features': 35} 0.874134529301  random_forest_sub1.csv
- 网络搜索的最优化参数：{'n_estimators':300,'min_samples_split':2,'max_feature':None, 'min_samples_leaf':1,'max_depth':15} 0.8668304 random_forest_sub2.csv

# 提升树模型（GradientBoostingRegressor）
loss: 选择损失函数，默认值为ls(least squres)
learning_rate: 学习率，模型是0.1
n_estimators: 弱学习器的数目，默认值100
max_depth: 每一个学习器的最大深度，限制回归树的节点数目，默认为3
min_samples_split: 可以划分为内部节点的最小样本数，默认为2
min_samples_leaf: 叶节点所需的最小样本数，默认为1
- 超参结果
- {'min_samples_split': 46, 'max_depth': 20, 'random_state': 1, 'n_estimators': 150} 0.890968226535 gradient_boost_sub1.csv
- 网络搜索的最优化参数：{'min_samples_split': 100, 'max_depth': 6, 'random_state': 1, 'n_estimators': 150} 0.890351143362 gradient_boost_sub2.csv


# ExtraTrees模型
- 默认参数：0.843513845948 submission/extra_tree_sub1.csv

# Ridge岭回归模型
- 网络搜索参数：alpha：10，fit_intercept:False, normalize:True 0.83986

# xgboost模型
- 默认参数：0.891816061168 xgboost_sub2.csv
- 运行网络搜索找到的最优参数{'learning_rate':0.1,'gamma':0.05,'max_depth':5,'min_child_weight':1,'subsample':0.6,'colsample_bytree':1.0} 0.898384 xgboost_sub1.csv











