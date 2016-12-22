# coding:utf-8
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    o_train = pd.read_csv('data/train.csv')
    o_test = pd.read_csv('data/test.csv')

    # 删除id列
    train = o_train.drop('Id',axis=1)
    test = o_test.drop('Id',axis=1)

    # 删除异常条目，有些居住面积大于4000平方英尺的数据显然有问题
    train.drop(train[train["GrLivArea"] > 4000].index, inplace=True)

    # 添加附加屬性
    train['HouseAge'] = train['YrSold'] - train['YearBuilt'] #房屋年齡
    train['RemodAge'] = train['YrSold'] - train['YearRemodAdd'] #重新裝修至今（年）
    test['HouseAge'] = test['YrSold'] - test['YearBuilt']  # 房屋年齡
    test['RemodAge'] = test['YrSold'] - test['YearRemodAdd']  # 重新裝修至今（年）

    # test中第66个条目，填充与车库有关的其他属性
    test.loc[666, "GarageQual"] = "TA"
    test.loc[666, "GarageCond"] = "TA"
    test.loc[666, "GarageFinish"] = "Unf"
    test.loc[666, "GarageYrBlt"] = "1980"

    # 假设1116没有车库
    test.loc[1116, "GarageType"] = np.nan
    # 均值填充LotFrontage
    lot_frontage_by_neighborhood = train["LotFrontage"].groupby(train["Neighborhood"])


    def munge(df):
        all_df = pd.DataFrame(index=df.index)

        all_df["LotFrontage"] = df["LotFrontage"]
        for key, group in lot_frontage_by_neighborhood:
            idx = (df["Neighborhood"] == key) & (df["LotFrontage"].isnull())
            all_df.loc[idx, "LotFrontage"] = group.median()

        all_df["LotArea"] = df["LotArea"]

        all_df["MasVnrArea"] = df["MasVnrArea"]
        all_df["MasVnrArea"].fillna(0, inplace=True)

        all_df["BsmtFinSF1"] = df["BsmtFinSF1"]
        all_df["BsmtFinSF1"].fillna(0, inplace=True)

        all_df["BsmtFinSF2"] = df["BsmtFinSF2"]
        all_df["BsmtFinSF2"].fillna(0, inplace=True)

        all_df["BsmtUnfSF"] = df["BsmtUnfSF"]
        all_df["BsmtUnfSF"].fillna(0, inplace=True)

        all_df["TotalBsmtSF"] = df["TotalBsmtSF"]
        all_df["TotalBsmtSF"].fillna(0, inplace=True)

        all_df["1stFlrSF"] = df["1stFlrSF"]
        all_df["2ndFlrSF"] = df["2ndFlrSF"]
        all_df["GrLivArea"] = df["GrLivArea"]

        all_df["GarageArea"] = df["GarageArea"]
        all_df["GarageArea"].fillna(0, inplace=True)

        all_df["WoodDeckSF"] = df["WoodDeckSF"]
        all_df["OpenPorchSF"] = df["OpenPorchSF"]
        all_df["EnclosedPorch"] = df["EnclosedPorch"]
        all_df["3SsnPorch"] = df["3SsnPorch"]
        all_df["ScreenPorch"] = df["ScreenPorch"]

        all_df["BsmtFullBath"] = df["BsmtFullBath"]
        all_df["BsmtFullBath"].fillna(0, inplace=True)

        all_df["BsmtHalfBath"] = df["BsmtHalfBath"]
        all_df["BsmtHalfBath"].fillna(0, inplace=True)

        all_df["FullBath"] = df["FullBath"]
        all_df["HalfBath"] = df["HalfBath"]
        all_df["BedroomAbvGr"] = df["BedroomAbvGr"]
        all_df["KitchenAbvGr"] = df["KitchenAbvGr"]
        all_df["TotRmsAbvGrd"] = df["TotRmsAbvGrd"]
        all_df["Fireplaces"] = df["Fireplaces"]

        all_df["GarageCars"] = df["GarageCars"]
        all_df["GarageCars"].fillna(0, inplace=True)

        all_df["CentralAir"] = (df["CentralAir"] == "Y") * 1.0

        all_df["OverallQual"] = df["OverallQual"]
        all_df["OverallCond"] = df["OverallCond"]

        # 将一些和质量相关的属性，可以对其进行量化表述，

        qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        all_df["ExterQual"] = df["ExterQual"].map(qual_dict).astype(int)
        all_df["ExterCond"] = df["ExterCond"].map(qual_dict).astype(int)
        all_df["BsmtQual"] = df["BsmtQual"].map(qual_dict).astype(int)
        all_df["BsmtCond"] = df["BsmtCond"].map(qual_dict).astype(int)
        all_df["HeatingQC"] = df["HeatingQC"].map(qual_dict).astype(int)
        all_df["KitchenQual"] = df["KitchenQual"].map(qual_dict).astype(int)
        all_df["FireplaceQu"] = df["FireplaceQu"].map(qual_dict).astype(int)
        all_df["GarageQual"] = df["GarageQual"].map(qual_dict).astype(int)
        all_df["GarageCond"] = df["GarageCond"].map(qual_dict).astype(int)

        all_df["BsmtExposure"] = df["BsmtExposure"].map(
            {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

        bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
        all_df["BsmtFinType1"] = df["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
        all_df["BsmtFinType2"] = df["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

        all_df["Functional"] = df["Functional"].map(
            {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
             "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

        all_df["GarageFinish"] = df["GarageFinish"].map(
            {None: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

        all_df["Fence"] = df["Fence"].map(
            {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)

        all_df["YearBuilt"] = df["YearBuilt"]
        all_df["YearRemodAdd"] = df["YearRemodAdd"]

        all_df["GarageYrBlt"] = df["GarageYrBlt"]
        all_df["GarageYrBlt"].fillna(0.0, inplace=True)

        all_df["MoSold"] = df["MoSold"]
        all_df["YrSold"] = df["YrSold"]

        all_df["LowQualFinSF"] = df["LowQualFinSF"]
        all_df["MiscVal"] = df["MiscVal"]

        all_df["PoolQC"] = df["PoolQC"].map(qual_dict).astype(int)

        all_df["PoolArea"] = df["PoolArea"]
        all_df["PoolArea"].fillna(0, inplace=True)

        # 对一些无序属性，采用
        all_df = factorize(df, all_df, "MSSubClass")
        all_df = factorize(df, all_df, "MSZoning", "RL")
        all_df = factorize(df, all_df, "LotConfig")
        all_df = factorize(df, all_df, "Neighborhood")
        all_df = factorize(df, all_df, "Condition1")
        all_df = factorize(df, all_df, "BldgType")
        all_df = factorize(df, all_df, "HouseStyle")
        all_df = factorize(df, all_df, "RoofStyle")
        all_df = factorize(df, all_df, "Exterior1st", "Other")
        all_df = factorize(df, all_df, "Exterior2nd", "Other")
        all_df = factorize(df, all_df, "MasVnrType", "None")
        all_df = factorize(df, all_df, "Foundation")
        all_df = factorize(df, all_df, "SaleType", "Oth")
        all_df = factorize(df, all_df, "SaleCondition")

        # IR2 and IR3 don't appear that often, so just make a distinction
        # between regular and irregular.
        all_df["IsRegularLotShape"] = (df["LotShape"] == "Reg") * 1

        # Most properties are level; bin the other possibilities together
        # as "not level".
        all_df["IsLandLevel"] = (df["LandContour"] == "Lvl") * 1

        # Most land slopes are gentle; treat the others as "not gentle".
        all_df["IsLandSlopeGentle"] = (df["LandSlope"] == "Gtl") * 1

        # Most properties use standard circuit breakers.
        all_df["IsElectricalSBrkr"] = (df["Electrical"] == "SBrkr") * 1

        # About 2/3rd have an attached garage.
        all_df["IsGarageDetached"] = (df["GarageType"] == "Detchd") * 1

        # Most have a paved drive. Treat dirt/gravel and partial pavement
        # as "not paved".
        all_df["IsPavedDrive"] = (df["PavedDrive"] == "Y") * 1

        # The only interesting "misc. feature" is the presence of a shed.
        all_df["HasShed"] = (df["MiscFeature"] == "Shed") * 1.

        # If YearRemodAdd != YearBuilt, then a remodeling took place at some point.
        all_df["Remodeled"] = (all_df["YearRemodAdd"] != all_df["YearBuilt"]) * 1

        # Did a remodeling happen in the year the house was sold?
        all_df["RecentRemodel"] = (all_df["YearRemodAdd"] == all_df["YrSold"]) * 1

        # Was this house sold in the year it was built?
        all_df["VeryNewHouse"] = (all_df["YearBuilt"] == all_df["YrSold"]) * 1

        all_df["Has2ndFloor"] = (all_df["2ndFlrSF"] == 0) * 1
        all_df["HasMasVnr"] = (all_df["MasVnrArea"] == 0) * 1
        all_df["HasWoodDeck"] = (all_df["WoodDeckSF"] == 0) * 1
        all_df["HasOpenPorch"] = (all_df["OpenPorchSF"] == 0) * 1
        all_df["HasEnclosedPorch"] = (all_df["EnclosedPorch"] == 0) * 1
        all_df["Has3SsnPorch"] = (all_df["3SsnPorch"] == 0) * 1
        all_df["HasScreenPorch"] = (all_df["ScreenPorch"] == 0) * 1

        # These features actually lower the score a little.
        # all_df["HasBasement"] = df["BsmtQual"].isnull() * 1
        # all_df["HasGarage"] = df["GarageQual"].isnull() * 1
        # all_df["HasFireplace"] = df["FireplaceQu"].isnull() * 1
        # all_df["HasFence"] = df["Fence"].isnull() * 1

        # Months with the largest number of deals may be significant.
        all_df["HighSeason"] = df["MoSold"].replace(
            {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

        all_df["NewerDwelling"] = df["MSSubClass"].replace(
            {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
             90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

        all_df.loc[df.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
        all_df.loc[df.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
        all_df.loc[df.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
        all_df.loc[df.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
        all_df.loc[df.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
        all_df["Neighborhood_Good"].fillna(0, inplace=True)

        all_df["SaleCondition_PriceDown"] = df.SaleCondition.replace(
            {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

        # House completed before sale or not
        all_df["BoughtOffPlan"] = df.SaleCondition.replace(
            {"Abnorml": 0, "Alloca": 0, "AdjLand": 0, "Family": 0, "Normal": 0, "Partial": 1})

        all_df["BadHeating"] = df.HeatingQC.replace(
            {'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})

        area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                     'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                     'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']
        all_df["TotalArea"] = all_df[area_cols].sum(axis=1)

        all_df["TotalArea1st2nd"] = all_df["1stFlrSF"] + all_df["2ndFlrSF"]

        all_df["Age"] = 2010 - all_df["YearBuilt"]
        all_df["TimeSinceSold"] = 2010 - all_df["YrSold"]

        all_df["SeasonSold"] = all_df["MoSold"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                                                     6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}).astype(int)

        all_df["YearsSinceRemodel"] = all_df["YrSold"] - all_df["YearRemodAdd"]

        # Simplifications of existing features into bad/average/good.
        all_df["SimplOverallQual"] = all_df.OverallQual.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
        all_df["SimplOverallCond"] = all_df.OverallCond.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
        all_df["SimplPoolQC"] = all_df.PoolQC.replace(
            {1: 1, 2: 1, 3: 2, 4: 2})
        all_df["SimplGarageCond"] = all_df.GarageCond.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplGarageQual"] = all_df.GarageQual.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplFireplaceQu"] = all_df.FireplaceQu.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplFireplaceQu"] = all_df.FireplaceQu.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplFunctional"] = all_df.Functional.replace(
            {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4})
        all_df["SimplKitchenQual"] = all_df.KitchenQual.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplHeatingQC"] = all_df.HeatingQC.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplBsmtFinType1"] = all_df.BsmtFinType1.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
        all_df["SimplBsmtFinType2"] = all_df.BsmtFinType2.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
        all_df["SimplBsmtCond"] = all_df.BsmtCond.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplBsmtQual"] = all_df.BsmtQual.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplExterCond"] = all_df.ExterCond.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        all_df["SimplExterQual"] = all_df.ExterQual.replace(
            {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})

        # Bin by neighborhood (a little arbitrarily). Values were computed by:
        # train_df["SalePrice"].groupby(train_df["Neighborhood"]).median().sort_values()
        neighborhood_map = {
            "MeadowV": 0,  # 88000
            "IDOTRR": 1,  # 103000
            "BrDale": 1,  # 106000
            "OldTown": 1,  # 119000
            "Edwards": 1,  # 119500
            "BrkSide": 1,  # 124300
            "Sawyer": 1,  # 135000
            "Blueste": 1,  # 137500
            "SWISU": 2,  # 139500
            "NAmes": 2,  # 140000
            "NPkVill": 2,  # 146000
            "Mitchel": 2,  # 153500
            "SawyerW": 2,  # 179900
            "Gilbert": 2,  # 181000
            "NWAmes": 2,  # 182900
            "Blmngtn": 2,  # 191000
            "CollgCr": 2,  # 197200
            "ClearCr": 3,  # 200250
            "Crawfor": 3,  # 200624
            "Veenker": 3,  # 218000
            "Somerst": 3,  # 225500
            "Timber": 3,  # 228475
            "StoneBr": 4,  # 278000
            "NoRidge": 4,  # 290000
            "NridgHt": 4,  # 315000
        }

        all_df["NeighborhoodBin"] = df["Neighborhood"].map(neighborhood_map)
        return all_df

    le = LabelEncoder()
    def factorize(df, factor_df, column, fill_na=None):
        '''
        将传入的列对应的object属性数字化，fill_na表示是否需要填充缺失值
        :param df:
        :param factor_df:
        :param column:
        :param fill_na:
        :return:
        '''
        factor_df[column] = df[column]
        if fill_na is not None:
            factor_df[column].fillna(fill_na, inplace=True)
        le.fit(factor_df[column].unique())
        factor_df[column] = le.transform(factor_df[column])
        return factor_df

    X_train = munge(train)
    X_test = munge(test)

    # Alley, PoolQC,Fence，FireplaceQu,MiscFeature大量缺失，用新的属性填充
    train['Alley'].fillna('NAN',inplace=True)
    train['PoolQC'].fillna('NAN', inplace=True)
    train['Fence'].fillna('NAN', inplace=True)
    train['FireplaceQu'].fillna('NAN', inplace=True)
    train['MiscFeature'].fillna('NAN', inplace=True)
    test['Alley'].fillna('NAN', inplace=True)
    test['PoolQC'].fillna('NAN', inplace=True)
    test['Fence'].fillna('NAN', inplace=True)
    test['FireplaceQu'].fillna('NAN', inplace=True)
    test['MiscFeature'].fillna('NAN', inplace=True)

    print(train.info())
    # 数据缺失值用均值代替
    train['LotFrontage'].fillna(train['LotFrontage'].mean(),inplace=True)
    train['MasVnrArea'].fillna(train['MasVnrArea'].mean(), inplace=True)
    test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace=True)
    test['MasVnrArea'].fillna(test['MasVnrArea'].mean(), inplace=True)
    test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean(), inplace=True)
    test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean(), inplace=True)
    test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean(), inplace=True)
    test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean(), inplace=True)
    test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean(), inplace=True)
    test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean(), inplace=True)
    test['GarageArea'].fillna(test['BsmtHalfBath'].mean(), inplace=True)
    test['GarageCars'].fillna(test['BsmtHalfBath'].mean(), inplace=True)
    test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean(), inplace=True)
    print(train.info())
    # 其余缺失值均用众数代替
    train.fillna(train.mode().loc[0], inplace=True)
    test.fillna(test.mode().loc[0], inplace=True)

    X_train = train.drop('SalePrice',axis=1)
    Y_train = train['SalePrice']
    # 向量化
    dict_vec = DictVectorizer(sparse=False)
    X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
    print(dict_vec.feature_names_)
    feature_name = dict_vec.feature_names_


    # select_feature = [0, 1, 2, 4, 5, 6, 7, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 33, 34, 35, 36, 38, 39, 40, 41, 42, 45, 65, 67, 68, 70, 71, 73, 74, 78, 80, 81, 83, 84, 87, 88, 95, 96, 98, 100, 103, 104, 110, 111, 113, 114, 116, 117, 118, 119, 120, 124, 129, 131, 132, 133, 139, 140, 141, 148, 150, 152, 153, 154, 155, 162, 163, 164, 166, 167, 168, 170, 173, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 193, 194, 195, 196, 198, 199, 200, 201, 204, 205, 206, 208, 209, 210, 217, 222, 223, 224, 225, 230, 233, 234, 235, 239, 240, 241, 243, 244, 245, 246, 248, 254, 262, 263, 265, 268, 271, 272, 273, 280, 282, 283, 286, 287, 290, 291, 292, 293]
    #
    # select_X_train = [[] for i in range(X_train.shape[0])]
    # for i in range(X_train.shape[0]):
    #     for j in select_feature:
    #         select_X_train[i].append(X_train[i][j])
    #
    # select_feature = np.array(select_X_train)
    # 向量化后，可能有些属性在test中从未出现，我们加上这些属性，并将对应列置0，填充test使之与train的维数相同

    # rfr = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=2, max_features=None,
    #                             min_samples_leaf=1, )
    gbr = GradientBoostingRegressor(n_estimators=150, max_depth=6, random_state=1, min_samples_split=100)
    # rfr.fit(X_train,Y_train)
    #

    # impts = rfr.feature_importances_
    # select_feature=[]
    # for a in range(impts.size):
    #     if impts[a]>= 0.1:
    #         select_feature.append(a)
    # print(select_feature)
    # 5折交叉验证
    valid_rfr = cross_val_score(gbr, X_train, Y_train, cv=5)
    print("RandomForestRegressor：", valid_rfr.mean())






