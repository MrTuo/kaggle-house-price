import pandas as pd
# 集成学习

if __name__ == "__main__":
    # 要就均值的文件
    ensemble_files = ['xgboost_sub1.csv',
                      'xgboost_sub2.csv',
                      'gradient_boost_sub2.csv',
                      'gradient_boost_sub1.csv',
                      ]
    s=[]
    for file in ensemble_files:
        f=pd.read_csv('submission/'+file)
        s.append(f['SalePrice'])

    rst = []
    for i in range(len(f['SalePrice'])):
        sum=0
        for j in range(len(ensemble_files)):
            sum += s[j][i]

        rst.append(sum/len(ensemble_files))

    submission = pd.DataFrame({'Id': f['Id'], 'SalePrice': rst})
    submission.to_csv('submission/avg_of_four.csv', index=False)


