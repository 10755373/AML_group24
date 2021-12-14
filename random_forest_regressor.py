from sklearn.ensemble import RandomForestRegressor
from functions import mse

class RandomForest:

    regressor = RandomForestRegressor(n_estimators=12, max_depth=8, random_state=42)

    def __init__(self, rf):
      self.regressor = rf

    def get_model(self):
        return self.regressor

    def train_model(self, X, y):
        self.regressor.fit(X,y)

    def predict(self, X):
        return self.regressor.predict(X)

    def evaluate_mse(self, y_actual, y_pred):
        return mse(y_actual, y_pred)
