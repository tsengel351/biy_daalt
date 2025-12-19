import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.get_logger().setLevel('ERROR')

class BaseModel:
    def __init__(self):
        self.scaler = StandardScaler()
    def _scale(self, X, fit=False):
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

class LR(BaseModel):
    """Logistic Regression"""
    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        super().__init__()
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, n_jobs=-1)
    def fit(self, X, y):
        self.model.fit(self._scale(X, True), y); return self
    def predict(self, X):
        return self.model.predict(self._scale(X))
    def predict_proba(self, X):
        return self.model.predict_proba(self._scale(X))

class AB(BaseModel):
    """AdaBoost"""
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        super().__init__()
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=random_state, algorithm="SAMME"
        )
    def fit(self, X, y):
        self.model.fit(self._scale(X, True), y); return self
    def predict(self, X):
        return self.model.predict(self._scale(X))
    def predict_proba(self, X):
        return self.model.predict_proba(self._scale(X))

class RF:
    """Random Forest"""
    def __init__(self, n_estimators=200, max_depth=20, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1)
    def fit(self, X, y):
        self.model.fit(X, y); return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LSTM(BaseModel):
    """LSTM (Keras)"""
    def __init__(self, units=128, dropout=0.3, epochs=10, batch_size=64, random_state=42):
        super().__init__()
        self.units = units; self.dropout = dropout
        self.epochs = epochs; self.batch_size = batch_size
        self.model = None
        tf.random.set_seed(random_state); np.random.seed(random_state)
    def fit(self, X, y):
        Xs = self._scale(X, True).reshape(-1, 1, X.shape[1])
        self.model = keras.Sequential([
            layers.Input(shape=(1, X.shape[1])),
            layers.LSTM(self.units, dropout=self.dropout),
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.fit(Xs, y, epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.1, verbose=0,
                       callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        return self
    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)
    def predict_proba(self, X):
        Xs = self._scale(X).reshape(-1, 1, X.shape[1])
        p = self.model.predict(Xs, verbose=0).flatten()
        return np.column_stack([1-p, p])
    def clear(self):
        keras.backend.clear_session()

def get_classifiers(seed=42):
    return {
        "Logistic Regression": {"class": LR, "params": {"random_state": seed}},
        "AdaBoost": {"class": AB, "params": {"random_state": seed}},
        "Random Forest": {"class": RF, "params": {"random_state": seed}},
        "LSTM": {"class": LSTM, "params": {"random_state": seed}},
    }