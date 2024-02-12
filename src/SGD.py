from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

from parseDataTEST import df

X = df[['beds', 'baths', 'size', 'lot_size', 'zip_code']]  
y = df['price']  # What you want to predict


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Define the model 
modelo_sgd = make_pipeline(StandardScaler(), linear_model.SGDRegressor(max_iter=15000, tol=1e-3))


param_grid = {
     'sgdregressor__alpha': [0.0001, 0.001, 0.01, 0.1],
    'sgdregressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
}
clf = GridSearchCV(modelo_sgd, param_grid)
clf.fit(X_train, y_train)
print("Best score: " + str(clf.best_score_))

# Entrenar el modelo
modelo_sgd.fit(X_train, y_train)

# Hacer predicciones sobre los datos de prueba
y_predicted = modelo_sgd.predict(X_test)
print("__________________________________________________________________________________________________________________")
print(y_predicted) #Predictions
print("__________________________________________________________________________________________________________________")

#Score of the predictions
score = modelo_sgd.score(X_train, y_train)
print("R-squared:", score)
