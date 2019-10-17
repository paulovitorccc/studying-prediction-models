import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as rmse, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


pd.set_option('display.max_columns', 100)

url = "https://github.com/paulovitorccc/studying-prediction-models/raw/master/data/cartola_2014.csv"
df = pd.read_csv(url, index_col=0)
df.head()

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

#x = df
#y = df.nota.values
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

train = train.fillna(0.0)
test = test.fillna(0.0)

x_train = df
y_train = x_train.nota.values
del x_train['nota']


hyperparams = {'alpha':[0.0005, 0.0014, 0.0006, 0.00061, 0.000612, 0.000613001, 0.000614, 0.00061401, 0.00061402, 0.00061403, 0.0006104 ]}
gs = GridSearchCV(estimator=Ridge(normalize=True), param_grid=hyperparams)
gs.fit(x_train, y_train)
pred = pd.Series(gs.predict(test))
err = gs.score(x_train, y_train)
print('Result:')
print('Best parameter: ',gs.best_params_)
print('Best score: ',gs.best_score_)
print('Root mean square logarithmic error: ', err)
print('\n')

ridge2 = Ridge(alpha = 0.0005, normalize=True)	
ridge2.fit(x_train, y_train)
print(ridge2.score(x_train, y_train))
result = pd.DataFrame(ridge2.predict(test), index = test.index, columns=['nota'])
print result
#result = result.drop_duplicates(subset='atleta_id', keep="last")
#result['atleta_id'] = result['atleta_id'].apply(lambda x:str(x))
result.to_csv('submission.csv')

