import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import *

plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})


# np.random.seed(4)

input_test = np.load('../../data/inconel_data/input_test_data.npy')
output_test = np.load('../../data/inconel_data/output_test_data.npy')


X_test = input_test
y_test = output_test

pca = joblib.load('../../models/inconel_models/inconel_pca.pkl')
model = joblib.load('../../models/inconel_models/inconel_model.pkl')



y_pred = pca.inverse_transform(model.predict(X_test))

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
plt.savefig('figa2a.pdf', dpi=300)

w = np.loadtxt('w_inconel.txt')
k = 43
plt.figure(figsize=(6, 5))
plt.plot(w, y_test[k], 'r-', label='Ground truth')
plt.plot(w, y_pred[k], 'g-',label=f'Prediction, RMSE = {np.sqrt(mean_squared_error(y_test[k], y_pred[k])):.3f}')
plt.legend()
ax = plt.gca()
plt.ylim(0, 1)
plt.xlim(2.5, 12)

plt.xlabel('Wavelength (µm)')
plt.ylabel('$\mathbf{\epsilon}$')
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
    
plt.savefig('figa2b.pdf', dpi=300)