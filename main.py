from mip_ml import add_predictor_constr
import mip
import numpy as np
from sklearn.linear_model import LinearRegression


# 学習
num_data = 10
X_train = np.random.randn(num_data, 1)
y_train = 3*X_train + 2 + np.random.randn(num_data)
reg = LinearRegression()
reg.fit(X_train, y_train)

# 最適化
m = mip.Model(solver_name=mip.CBC)
x = m.add_var('x', lb=0, ub=10)
y = m.add_var('y')
add_predictor_constr(m, reg, x, y)
m.objective = mip.maximize(x)
m.optimize()
