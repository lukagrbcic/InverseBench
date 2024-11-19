import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
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

# np.save('input_test_data.npy', X_test)
# np.save('output_test_data.npy', y_test)



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


def calculate_rmse(y_true, y_pred):   
    rmse = np.mean(np.array([np.sqrt(mean_squared_error(y_true[i], y_pred[i])) for i in range(len(y_pred))]))
    return rmse


y_pred = model.predict(X_test)

rmse_i = np.array([np.sqrt(mean_squared_error(y_test[i], y_pred[i])) for i in range(len(y_pred))])
r2_i = np.array([r2_score(y_test[i], y_pred[i]) for i in range(len(y_pred))])


rmse_single = np.sqrt(mean_squared_error(y_test, y_pred))
r2_single = r2_score(y_test, y_pred)


print (rmse_single)
print (r2_single)


plt.figure(figsize=(6, 5))
plt.hist(rmse_i, bins=50, color='green')
plt.xlabel('RMSE')
plt.ylabel('Predictions')
# plt.title('XGBoost RMSE Over Epochs')
# plt.legend()
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

plt.tight_layout()
plt.savefig('figa3a.pdf', dpi=300)


results = model.evals_result()
training_errors = results['validation_0']['rmse']
validation_errors = results['validation_1']['rmse']

epochs = len(training_errors)
x_axis = range(0, epochs)

plt.figure(figsize=(6, 5))
plt.plot(x_axis, training_errors, label='Train', color='red')
plt.plot(x_axis, validation_errors, label='Validation', color='green')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
# plt.title('XGBoost RMSE Over Epochs')
plt.legend()
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

plt.tight_layout()
plt.savefig('figa3b.pdf', dpi=300)

    
k = 1
plt.figure(figsize=(6, 5))
plt.plot(np.arange(0, len(y_test[k]), 1), y_test[k], 'ro', label='Ground truth')
plt.plot(np.arange(0, len(y_pred[k]), 1), y_pred[k], color='green', marker='o', 
         linestyle='',label=f'Prediction, RMSE = {np.sqrt(mean_squared_error(y_test[k], y_pred[k])):.3f}')
plt.legend()
ax = plt.gca()
plt.ylim(0,30)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

plt.tight_layout()
plt.ylabel('$\mathbf{c}$')
plt.xlabel('Measurement locations')
    
plt.savefig('figa3c.pdf', dpi=300)
# # joblib.dump(model, 'scalar_diffusion_model.pkl')
# def calculate_rmse(y_true, y_pred):   
#     # rmse = np.mean(np.array([np.sqrt(mean_squared_error(y_true[i], y_pred[i])) for i in range(len(y_pred))]))
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     return rmse


# def calculate_mape(y_true, y_pred):   
#     # mape = np.mean(np.array([mean_absolute_percentage_error(y_true[i], y_pred[i]) for i in range(len(y_pred))]))
#     mape = mean_absolute_percentage_error(y_true, y_pred)
#     return mape


# # y_val_pred = model.predict(X_val)
# # val_rmse_per_target = calculate_rmse(y_val, y_val_pred)
# # print("Validation RMSE per target:", val_rmse_per_target)
# # print("Average Validation RMSE:", np.mean(val_rmse_per_target))

# y_test_pred = model.predict(X_test)
# test_rmse_per_target = calculate_rmse(y_test, y_test_pred)
# print("\nTest RMSE per target:", test_rmse_per_target)
# print("Average Test RMSE:", np.mean(test_rmse_per_target))


# test_mape_per_target = calculate_mape(y_test, y_test_pred)
# print("\nTest MAPE per target:", test_mape_per_target)
# print("Average Test MAPE:", np.mean(test_mape_per_target))

# from sklearn.metrics import r2_score
# print ('R2', r2_score(y_test, y_test_pred))

# for i in range(20):
    
    
    # plt.figure()
    # plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test[i], 'r-')
    # plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test_pred[i], 'g-')
    
#     plt.figure()
#     plt.plot(np.arange(0, len(X_test[i]), 1), X_test[i], 'bo')
#     # plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test_pred[i], 'bo')
    
    
    
#     # plt.figure()
#     # plt.title('mpt')
#     # plt.plot(np.arange(0, len(y_test_pred[i,2:]), 1), y_test[i, 2:], 'ro')
#     # plt.plot(np.arange(0, len(y_test_pred[i,2:]), 1), y_test_pred[i, 2:], 'go')
    
#     # plt.figure()
#     # plt.title('Re')
#     # plt.plot(y_test[i, 0], 'ro')
#     # plt.plot(y_test_pred[i, 0], 'go')
#     # # plt.ylim(0, 1)
    
#     # plt.figure()
#     # plt.title('AoA')
#     # plt.plot(y_test[i, 1], 'ro')
#     # plt.plot(y_test_pred[i, 1], 'go')
    
    
    
#     # plt.figure()
#     # plt.title('mpt')
#     # plt.plot(np.arange(0, len(y_test_pred[i,1:]), 1), y_test[i, 1:], 'ro')
#     # plt.plot(np.arange(0, len(y_test_pred[i,1:]), 1), y_test_pred[i, 1:], 'go')
    
#     # plt.figure()
#     # plt.title('Re')
#     # plt.plot(y_test[i, 0], 'ro')
#     # plt.plot(y_test_pred[i, 0], 'go')
  


# # val_scores = model.eval_set([(dval, 'val')])

# # # # Extract RMSE values and convert to a list
# # val_rmse = [float(score.split(':')[1]) for score in val_scores]

# # # Plot the validation loss
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(1, len(val_rmse) + 1), val_rmse)
# # plt.title('XGBoost Validation RMSE over Iterations')
# # plt.xlabel('Iterations')
# # plt.ylabel('Validation RMSE')
# # plt.show()


