import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib

# np.random.seed(4)

# probes = np.loadtxt('../../data/scalar_diffusion_data/probes.txt')#[:500]
# bcs = np.loadtxt('../../data/scalar_diffusion_data/bcs.txt')#[:500]


probes = np.loadtxt('probes_30.txt')#[:500]
bcs = np.loadtxt('bcs_30.txt')#[:500]


print(np.min(bcs, axis=0))
print (np.max(bcs, axis=0))



# print(np.min(probes, axis=0))
# print (np.max(probes, axis=0))

X = bcs
y = probes


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=23)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=23)

np.save('input_test_data.npy', X_test)
np.save('output_test_data.npy', y_test)



# # print(np.min(X_train, axis=0))
# # print (np.max(X_train, axis=0))

# import sys
# sys.exit()


# model = RandomForestRegressor(n_estimators=170).fit(X_train, y_train)
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    learning_rate=0.1,
    max_depth=3,
    # subsample=0.8,
    # colsample_bytree=0.8,
    reg_alpha=20,
    reg_lambda=50,
    n_estimators=3000
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=5,
    verbose=True
)

joblib.dump(model, 'scalar_diffusion_model.pkl')
def calculate_rmse(y_true, y_pred):   
    # rmse = np.mean(np.array([np.sqrt(mean_squared_error(y_true[i], y_pred[i])) for i in range(len(y_pred))]))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def calculate_mape(y_true, y_pred):   
    # mape = np.mean(np.array([mean_absolute_percentage_error(y_true[i], y_pred[i]) for i in range(len(y_pred))]))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mape


# y_val_pred = model.predict(X_val)
# val_rmse_per_target = calculate_rmse(y_val, y_val_pred)
# print("Validation RMSE per target:", val_rmse_per_target)
# print("Average Validation RMSE:", np.mean(val_rmse_per_target))

y_test_pred = model.predict(X_test)
test_rmse_per_target = calculate_rmse(y_test, y_test_pred)
print("\nTest RMSE per target:", test_rmse_per_target)
print("Average Test RMSE:", np.mean(test_rmse_per_target))


test_mape_per_target = calculate_mape(y_test, y_test_pred)
print("\nTest MAPE per target:", test_mape_per_target)
print("Average Test MAPE:", np.mean(test_mape_per_target))

from sklearn.metrics import r2_score
print ('R2', r2_score(y_test, y_test_pred))

for i in range(20):
    
    
    # plt.figure()
    # plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test[i], 'r-')
    # plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test_pred[i], 'g-')
    
    plt.figure()
    plt.plot(np.arange(0, len(X_test[i]), 1), X_test[i], 'bo')
    # plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test_pred[i], 'bo')
    
    
    
    # plt.figure()
    # plt.title('mpt')
    # plt.plot(np.arange(0, len(y_test_pred[i,2:]), 1), y_test[i, 2:], 'ro')
    # plt.plot(np.arange(0, len(y_test_pred[i,2:]), 1), y_test_pred[i, 2:], 'go')
    
    # plt.figure()
    # plt.title('Re')
    # plt.plot(y_test[i, 0], 'ro')
    # plt.plot(y_test_pred[i, 0], 'go')
    # # plt.ylim(0, 1)
    
    # plt.figure()
    # plt.title('AoA')
    # plt.plot(y_test[i, 1], 'ro')
    # plt.plot(y_test_pred[i, 1], 'go')
    
    
    
    # plt.figure()
    # plt.title('mpt')
    # plt.plot(np.arange(0, len(y_test_pred[i,1:]), 1), y_test[i, 1:], 'ro')
    # plt.plot(np.arange(0, len(y_test_pred[i,1:]), 1), y_test_pred[i, 1:], 'go')
    
    # plt.figure()
    # plt.title('Re')
    # plt.plot(y_test[i, 0], 'ro')
    # plt.plot(y_test_pred[i, 0], 'go')
  


# val_scores = model.eval_set([(dval, 'val')])

# # # Extract RMSE values and convert to a list
# val_rmse = [float(score.split(':')[1]) for score in val_scores]

# # Plot the validation loss
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(val_rmse) + 1), val_rmse)
# plt.title('XGBoost Validation RMSE over Iterations')
# plt.xlabel('Iterations')
# plt.ylabel('Validation RMSE')
# plt.show()


