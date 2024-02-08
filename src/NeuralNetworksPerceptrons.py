import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from parseDataTEST import df
from sklearn.neural_network import MLPClassifier


##A view of the data and were are they positioned 


X = df[['beds', 'baths', 'size','lot_size','zip_code']]  # Select the columns you want as features
y = df['price']  # Select the column containing your target labels


X = StandardScaler().fit_transform(X)

print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)




print("hello")

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test, cmap=cm_bright, alpha=0.3)

plt.show()




clf = MLPClassifier(hidden_layer_sizes=(1,), solver='sgd', 
                    batch_size=4, learning_rate_init=0.005,
                    max_iter=500, shuffle=True)
# Train the MLP classifier on training dataset
clf.fit(X_train, y_train)
print("Number of layers: ", clf.n_layers_)
print("he llegao")
print("Number of outputs: ", clf.n_outputs_)



h = np.argmax(clf.predict(X_train))
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
ax[0].set_title("Data")
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=h, cmap=cm_bright)
ax[1].set_title("Prediction")