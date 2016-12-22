# coding:utf-8
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    # 添加附加屬性
    train['HouseAge'] = train['YrSold'] - train['YearBuilt'] #房屋年齡
    train['RemodAge'] = train['YrSold'] - train['YearRemodAdd'] #重新裝修至今（年）
    test['HouseAge'] = test['YrSold'] - test['YearBuilt']  # 房屋年齡
    test['RemodAge'] = test['YrSold'] - test['YearRemodAdd']  # 重新裝修至今（年）
    # print(train.info())
    # 选择特征值
    selected_features = ['GrLivArea','TotalBsmtSF','OverallQual','1stFlrSF','LotArea','2ndFlrSF','BsmtFinSF1','GarageArea',
                         'GarageCars','YearBuilt','TotRmsAbvGrd','MSSubClass','Fireplaces','FullBath','FireplaceQu','ExterQual',
                         'Neighborhood','YearRemodAdd','OverallCond','KitchenQual','MSZoning','BedroomAbvGr','GarageType',
                         'BldgType','HalfBath','MasVnrArea','BsmtQual','GarageFinish','OpenPorchSF','HouseStyle','BsmtUnfSF',
                         'BsmtFinType1','LotFrontage','Foundation','HeatingQC','WoodDeckSF','CentralAir','GarageQual','BsmtCond',
                         'BsmtExposure','Exterior1st','KitchenAbvGr','Exterior2nd','LotShape','RoofStyle','MasVnrType','BsmtFinType2',
                         'PavedDrive','ScreenPorch','Functional','LandContour','Condition1','SaleCondition','Electrical','LandSlope',
                         'HouseAge','RemodAge',
                         'Fence',
                         ]
    X_train = train[selected_features]
    X_test = test[selected_features]
    Y_train = train['SalePrice']

    # 填充缺失值
    fill_NA = 'NA'
    NA_arr = ['FireplaceQu','GarageType','BsmtQual','GarageFinish','BsmtFinType1','GarageQual','BsmtCond','BsmtExposure',
              'BsmtFinType2',
              'Fence',
              ]
    for NA_atr in NA_arr:
        X_train[NA_atr].fillna(X_train[NA_atr].mode().loc[0], inplace=True)
        X_test[NA_atr].fillna(X_train[NA_atr].mode().loc[0], inplace=True)
    # X_train['FireplaceQu'].fillna(fill_NA,inplace=True)
    # X_train['GarageType'].fillna(fill_NA,inplace=True)
    # X_train['BsmtQual'].fillna(fill_NA,inplace=True)
    # X_train['GarageFinish'].fillna(fill_NA,inplace=True)
    # X_train['BsmtFinType1'].fillna(fill_NA,inplace=True)
    # X_train['GarageQual'].fillna(fill_NA, inplace=True)
    # X_train['BsmtCond'].fillna(fill_NA, inplace=True)
    # X_train['BsmtExposure'].fillna(fill_NA,inplace=True)
    # X_train['BsmtFinType2'].fillna(fill_NA,inplace=True)
    # X_train['Fence'].fillna(fill_NA, inplace=True)

    X_train['LotFrontage'].fillna(X_train['LotFrontage'].mean(), inplace=True)
    X_train['MasVnrArea'].fillna(0, inplace=True)
    X_train['MasVnrType'].fillna(X_train['MasVnrType'].mode().loc[0], inplace=True)
    X_train['Electrical'].fillna(X_train['Electrical'].mode().loc[0], inplace = True)

    print(X_train.info())
    # X_test['FireplaceQu'].fillna(fill_NA, inplace=True)
    # X_test['GarageType'].fillna(fill_NA, inplace=True)
    # X_test['BsmtQual'].fillna(fill_NA, inplace=True)
    # X_test['GarageFinish'].fillna(fill_NA, inplace=True)
    # X_test['BsmtFinType1'].fillna(fill_NA, inplace=True)
    # X_test['GarageQual'].fillna(fill_NA, inplace=True)
    # X_test['BsmtCond'].fillna(fill_NA, inplace=True)
    # X_test['BsmtExposure'].fillna(fill_NA, inplace=True)
    # X_test['BsmtFinType2'].fillna(fill_NA, inplace=True)
    # X_test['Fence'].fillna(fill_NA, inplace=True)

    X_test['MasVnrArea'].fillna(0, inplace=True)
    X_test['LotFrontage'].fillna(X_train['LotFrontage'].mean(), inplace=True)
    X_test['BsmtUnfSF'].fillna(X_train['BsmtUnfSF'].mean(), inplace=True)
    X_test['MasVnrType'].fillna(X_train['MasVnrType'].mode().loc[0], inplace=True)
    X_test['Electrical'].fillna(X_train['Electrical'].mode().loc[0], inplace=True)
    X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean(), inplace=True)
    X_test['BsmtFinSF1'].fillna(X_test['BsmtFinSF1'].mean(), inplace=True)
    X_test['GarageArea'].fillna(X_test['GarageArea'].mean(), inplace=True)
    X_test['GarageCars'].fillna(X_test['GarageCars'].mean(), inplace=True)
    X_test['KitchenQual'].fillna('Po', inplace=True)
    X_test['MSZoning'].fillna(X_test['MSZoning'].mode().loc[0], inplace=True) # 出现次数最多的值
    X_test['Exterior1st'].fillna(X_test['Exterior1st'].mode().loc[0], inplace=True) #
    X_test['Exterior2nd'].fillna(X_test['Exterior2nd'].mode().loc[0], inplace=True)
    X_test['Functional'].fillna(X_test['Functional'].mode().loc[0], inplace=True)

    print(X_test.info())
    # 特征向量化
    dict_vec=DictVectorizer(sparse=False)
    X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
    print(dict_vec.feature_names_)
    X_test = dict_vec.fit_transform(X_test.to_dict(orient='record'))
    print(dict_vec.feature_names_)
    # X_test 48維Electrical=Mix缺失，
    #        61維Exterior1st=ImStucc缺失，
    #        64維Exterior1st=Stone缺失，
    #        78維Exterior2nd=Other缺失，
    #        114维 'GarageQual=Ex'缺失，用0补齐
    X_test_arr=X_test.tolist()
    for i in range(len(X_test)):
        X_test_arr[i].insert(114,0)
        X_test_arr[i].insert(78, 0)
        X_test_arr[i].insert(64, 0)
        X_test_arr[i].insert(61, 0)
        X_test_arr[i].insert(48, 0)
    X_test = np.array(X_test_arr)


    # 交叉验证的岭回归
    # lr = linear_model.RidgeCV(alphas=[1.0, 10.0,100])
    # 随机森林回归 不同参数选择

    # rfr = RandomForestRegressor(n_estimators=150, max_depth=25, max_features=30)
    # # 回归决策树
    # dtr = DecisionTreeRegressor()
    etr = ExtraTreesRegressor()
    # gbr = GradientBoostingRegressor(n_estimators=150,max_depth=30,random_state=1,min_samples_split=40)
    # gbr = GradientBoostingRegressor(n_estimators=150, max_depth=20, random_state=1, min_samples_split=46)
    rfr = RandomForestRegressor()
    rv = Ridge()
    lr = LinearRegression()
    xgb = XGBRegressor()

    # 採用網絡搜索找到更好地超參組合
    # params = {'n_estimators': [150], 'max_depth': list(range(15, 45, 5)),'random_state': [1],
    #           'min_samples_split': list(range(30,50,2))}
    rfr_params={'n_estimators':[120,150,300,500,800,1200],'max_depth':[5,8,15,20,30,None],'max_features':['log2','sqrt',None],'min_samples_split':[1,2,5,10,15,100],'min_samples_leaf':[1,2,5,10]}
    rv_params = {'alpha':[0.01,0.1,1,10,100,1000],'fit_intercept':[True,False],'normalize':[True,False]}
    lr_params = {'fit_intercept':[True,False],'normalize':[True,False]}
    xgb_params = {'learning_rate':[0.01,0.025,0.1,0.2],'gamma':[0.05,0.3,0.7,1.0],'max_depth':[3,5,10,None],'min_child_weight':[1,4,7],'subsample':[0.6,0.8,1.0],'colsample_bytree':[0.6,0.8,1.0]}


    # #gbr = GradientBoostingRegressor()
    # gbr = RandomForestRegressor()
    # gs = GridSearchCV(gbr,params,n_jobs=-1,cv=5,verbose=1)
    # gs.fit(X_train,Y_train)
    # print(gs.best_params_)
    # print(gs.best_score_)

    # rv_gs = GridSearchCV(rv, rv_params, cv=5, verbose=1)
    # rv_gs.fit(X_train, Y_train)
    # print("rv", rv_gs.best_params_)
    # print("rv", rv_gs.best_score_)
    #
    # lr_gs = GridSearchCV(lr, lr_params, cv=5, verbose=1)
    # lr_gs.fit(X_train, Y_train)
    # print("lr", lr_gs.best_params_)
    # print("lr", lr_gs.best_score_)
    #
    # rfr_gs = GridSearchCV(rfr, rfr_params, n_jobs=-1, cv=5, verbose=1)
    # rfr_gs.fit(X_train, Y_train)
    # print("rfr", rfr_gs.best_params_)
    # print("rfr", rfr_gs.best_score_)

    # rfr_xgb = GridSearchCV(xgb, xgb_params, n_jobs=-1, cv=5, verbose=1)
    # rfr_xgb.fit(X_train, Y_train)
    # print("xgb", rfr_xgb.best_params_)
    # print("xgb", rfr_xgb.best_score_)

    # 5折交叉验证法，验证准确率
    # xgb = XGBRegressor(learning_rate=0.1, gamma=0.05, max_depth=5, min_child_weight=1, subsample=0.6,
    #                    colsample_bytree=1.0, )
    # rfr = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=2,max_features=None,min_samples_leaf=1,)
    gbr = GradientBoostingRegressor(n_estimators=150, max_depth=6, random_state=1, min_samples_split=100)
    # valid_lr = cross_val_score(lr,X_train,Y_train,cv=5)
    # valid_rfr = cross_val_score(rfr,X_train,Y_train,cv=5)
    # valid_dtr = cross_val_score(dtr,X_train,Y_train,cv=5)
    # valid_etr = cross_val_score(etr, X_train, Y_train, cv=5)
    # valid_gbr = cross_val_score(gbr, X_train, Y_train, cv=5)
    # valid_rv = cross_val_score(rv, X_train, Y_train, cv=5)
    # valid_lr = cross_val_score(lr, X_train, Y_train, cv=5)
    # valid_xgb = cross_val_score(xgb, X_train, Y_train, cv=5)
    # print("RidgeCV:",valid_lr.mean())
    # print("RandomForestRegressor：",valid_rfr.mean())
    # print("DecisionTreeRegressor：", valid_dtr.mean())
    # print("ExtraTreesRegressor：", valid_etr.mean())
    # print("GradientBoostingRegressor：", valid_gbr.mean())
    # print("Ridge：", valid_rv.mean())
    # print("lr:",valid_lr.mean())
    # print("xgb:",valid_xgb.mean())

    # 预测
    gbr.fit(X_train,Y_train)
    rst = gbr.predict(X_test)
    submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': rst})
    submission.to_csv('submission/', index=False)




