import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

data = pd.read_csv('covid.train.csv')
x = data[data.columns[1:94]]
y = data[data.columns[94]]
x = (x - x.min()) / (x.max() - x.min())

# fit
bestfeatures = SelectKBest(score_func=f_regression, k=5)
fit = bestfeatures.fit(x,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(15,'Score'))  #print 15 best features