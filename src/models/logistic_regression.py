"""
Logistic Regression загвар
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from typing import Dict, Any
import joblib
import config


class LogisticRegressionModel:
    """
    Logistic Regression классификатор
    
    Математик үндэслэл:
    Logistic Regression нь бинар ангиллын сонгодог загвар бөгөөд 
    sigmoid функц ашиглан магадлал тооцдог: 
    
    P(y=1|x) = 1 / (1 + exp(-(w^T * x + b)))
    
    Зорилтот функц (Log Loss):
    L(w) = -1/N * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
    
    Үүнийг L1 (Lasso) эсвэл L2 (Ridge) regularization-тай хослуулан 
    overfitting-ээс сэргийлнэ. 
    """
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              use_cv: bool = True) -> Dict[str, Any]:
        """
        Загварыг сургах
        
        Args:
            X_train: Сургалтын онцлогууд
            y_train:  Сургалтын labels
            use_cv: Cross-validation ашиглах эсэх
            
        Returns:
            Сургалтын мэдээлэл
        """
        print("Training Logistic Regression...")
        
        if use_cv:
            # Cross-validation тохируулга
            cv = RepeatedStratifiedKFold(
                n_splits=config.CV_N_SPLITS,
                n_repeats=config.CV_N_REPEATS,
                random_state=config. RANDOM_STATE
            )
            
            # Grid search
            grid_search = GridSearchCV(
                LogisticRegression(random_state=config.RANDOM_STATE),
                param_grid=config. CLASSIFIER_PARAMS['logistic_regression'],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search. fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self. best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return {
                'best_params': self.best_params,
                'best_cv_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        else: 
            # CV-гүйгээр энгийн сургалт
            self.model = LogisticRegression(random_state=config.RANDOM_STATE)
            self.model. fit(X_train, y_train)
            
            return {'trained': True}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Урьдчилсан таамаглал"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np. ndarray) -> np.ndarray:
        """Магадлалын урьдчилсан таамаглал"""
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Загварыг хадгалах"""
        joblib.dump(self.model, filepath)
        print(f"Logistic Regression model saved to {filepath}")
        
    def load(self, filepath: str):
        """Загварыг ачаалах"""
        self.model = joblib.load(filepath)
        print(f"Logistic Regression model loaded from {filepath}")