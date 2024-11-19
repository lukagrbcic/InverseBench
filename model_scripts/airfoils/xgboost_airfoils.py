import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import joblib


plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})


np.random.seed(4)

mpt = np.load('../../data/airfoil_data/input_data.npy')
cp = np.load('../../data/airfoil_data/output_data.npy')[:, ::4]

X = mpt
y = cp

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=23)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=23)






# np.save('input_train_data.npy', X_train)
# np.save('output_train_data.npy', y_train)


# np.save('input_test_data.npy', X_test)
# np.save('output_test_data.npy', y_test)

# import sys
# sys.exit()




dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# params = {
#     'objective': 'reg:squarederror',
#     'eval_metric': 'rmse',
#     'eta': 0.1,
#     'max_depth': 3,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'reg_lamdba': 10
# }

# num_rounds = 2000
# model = xgb.train(
#     params,
#     dtrain,
#     num_rounds,
#     evals=[(dtrain, 'train'), (dval, 'val')],
#     early_stopping_rounds=10,
#     verbose_eval=1
# )

# Create the XGBRegressor model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    eta=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,
    n_estimators=2000
)

# Fit the model with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=10,
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
plt.savefig('figa1a.pdf', dpi=300)


# plt.figure(figsize=(6, 5))
# plt.hist(r2_i, bins=50, color='green')
# plt.xlabel('R$^2$')
# plt.ylabel('Predictions')
# plt.xlim(0.9, 1)
# # plt.title('XGBoost RMSE Over Epochs')
# # plt.legend()
# ax = plt.gca()
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(2)

# plt.tight_layout()
# plt.savefig('airfoil_r2_bins.pdf', dpi=300)



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
plt.savefig('figa1b.pdf', dpi=300)




# k = 1
# plt.figure(figsize=(6, 5))


# y_test_half = y_test[k][:int(0.5*len(y_test))]
# y_test_half_2 = y_test[k][int(0.5*len(y_test)):]


# plt.plot(np.arange(0, len(y_test[k]), 1), y_test[k], 'r-', label='Ground truth')
# # plt.plot(np.arange(0, len(y_pred[k]), 1), y_pred[k], 'g-', label=f'Prediction, RMSE = {np.sqrt(np.sqrt(mean_squared_error(y_test[k], y_pred[k]))):.3f}')
# plt.legend()
# ax = plt.gca()
# # plt.ylim(0,30)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(2)

# plt.tight_layout()
    
# plt.savefig('figa1c.pdf', dpi=300)




k = 1208
plt.figure(figsize=(6, 5))


y_test_half = y_test[k][:int(0.5*len(y_test[k]))+1]
y_test_half_2 = y_test[k][int(0.5*len(y_test[k])):]


plt.plot(np.linspace(0, 1, len(y_test_half)), y_test_half[::-1], 'r-', label='Ground truth')
plt.plot(np.linspace(0, 1, len(y_test_half_2)), y_test_half_2, 'r-')



y_pred_half = y_pred[k][:int(0.5*len(y_pred[k]))+1]
y_pred_half_2 = y_pred[k][int(0.5*len(y_pred[k])):]


plt.plot(np.linspace(0, 1, len(y_pred_half)), y_test_half[::-1], 'g-', label=f'Prediction, RMSE = {np.sqrt(mean_squared_error(y_test[k], y_pred[k])):.3f}')
plt.plot(np.linspace(0, 1, len(y_pred_half_2)), y_pred_half_2, 'g-')


# plt.plot(np.arange(0, len(y_pred[k]), 1), y_pred[k], 'g-', label=f'Prediction, RMSE = {np.sqrt(np.sqrt(mean_squared_error(y_test[k], y_pred[k]))):.3f}')
plt.legend()
ax = plt.gca()
# plt.ylim(0,30)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

# plt.tight_layout()
    
plt.xlabel('$\zeta_x$')
plt.ylabel('$\mathbf{C_p}$')

plt.savefig('figa1c.pdf', dpi=300)



# y_test_pred = model.predict(X_test)


# rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))


# print (rmse)

# joblib.dump(model, 'airfoil_model.pkl')


# def calculate_rmse(y_true, y_pred):   
#     rmse = np.mean(np.array([np.sqrt(mean_squared_error(y_true[i], y_pred[i])) for i in range(len(y_pred))]))
#     return rmse


# def calculate_mape(y_true, y_pred):   
#     mape = np.mean(np.array([mean_absolute_percentage_error(y_true[i], y_pred[i]) for i in range(len(y_pred))]))
#     return mape


# # Evaluate on validation set
# y_val_pred = model.predict(X_val)
# val_rmse_per_target = calculate_rmse(y_val, y_val_pred)
# print("Validation RMSE per target:", val_rmse_per_target)
# print("Average Validation RMSE:", np.mean(val_rmse_per_target))

# Evaluate on test set
# y_test_pred = model.predict(X_test)
# # test_rmse_per_target = calculate_rmse(y_test, y_test_pred)
# print("\nTest RMSE per target:", test_rmse_per_target)
# print("Average Test RMSE:", np.mean(test_rmse_per_target))


# # Evaluate on test set
# y_test_pred = model.predict(X_test)
# test_rmse_per_target = calculate_rmse(y_test, y_test_pred)
# print("\nTest RMSE per target:", test_rmse_per_target)
# print("Average Test RMSE:", np.mean(test_rmse_per_target))



# Evaluate on test set
# y_test_pred = model.predict(dtest)
# test_mape_per_target = calculate_mape(y_test, y_test_pred)
# print("\nTest MAPE per target:", test_mape_per_target)
# print("Average Test MAPE:", np.mean(test_mape_per_target))

#TODO ADD PLOTS!

# for i in range(20):
    
    
#     plt.figure()
#     plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test[i], 'r-')
#     plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test_pred[i], 'g-')
    
    
#     plt.figure()
#     plt.title('mpt')
#     plt.plot(np.arange(0, len(y_test_pred[i,2:]), 1), y_test[i, 2:], 'ro')
#     plt.plot(np.arange(0, len(y_test_pred[i,2:]), 1), y_test_pred[i, 2:], 'go')
    
#     plt.figure()
#     plt.title('Re')
#     plt.plot(y_test[i, 0], 'ro')
#     plt.plot(y_test_pred[i, 0], 'go')
#     # plt.ylim(0, 1)
    
#     plt.figure()
#     plt.title('AoA')
#     plt.plot(y_test[i, 1], 'ro')
#     plt.plot(y_test_pred[i, 1], 'go')
    
    
    
#     # plt.figure()
#     # plt.title('mpt')
#     # plt.plot(np.arange(0, len(y_test_pred[i,1:]), 1), y_test[i, 1:], 'ro')
#     # plt.plot(np.arange(0, len(y_test_pred[i,1:]), 1), y_test_pred[i, 1:], 'go')
    
#     # plt.figure()
#     # plt.title('Re')
#     # plt.plot(y_test[i, 0], 'ro')
#     # plt.plot(y_test_pred[i, 0], 'go')
  



