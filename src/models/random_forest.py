"""
Random Forest загвар
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from typing import Dict, Any
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


class RandomForestModel:
    """Random Forest классификатор"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              use_cv: bool = True) -> Dict[str, Any]:
        """Загварыг сургах"""
        print("Training Random Forest...")
        
        if use_cv:
            cv = RepeatedStratifiedKFold(
                n_splits=config.CV_N_SPLITS,
                n_repeats=config.CV_N_REPEATS,
                random_state=config.RANDOM_STATE
            )
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=config. RANDOM_STATE),
                param_grid=config.CLASSIFIER_PARAMS['random_forest'],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search. best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters:  {self.best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return {
                'best_params': self.best_params,
                'best_cv_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_,
                'feature_importances': self.model.feature_importances_
            }
        else: 
            self.model = RandomForestClassifier(random_state=config.RANDOM_STATE)
            self.model.fit(X_train, y_train)
            return {
                'trained': True,
                'feature_importances':  self.model.feature_importances_
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Урьдчилсан таамаглал"""
        return self. model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Магадлалын урьдчилсан таамаглал"""
        return self.model.predict_proba(X)
    
    def get_feature_importances(self) -> np.ndarray:
        """Онцлогуудын ач холбогдлыг авах"""
        return self.model.feature_importances_
    
    def save(self, filepath: str):
        """Загварыг хадгалах"""
        joblib.dump(self.model, filepath)
        print(f"Random Forest model saved to {filepath}")
        
    def load(self, filepath: str):
        """Загварыг ачаалах"""
        self.model = joblib. load(filepath)
        print(f"Random Forest model loaded from {filepath}")


if __name__ == "__main__":
    print("✓ Random Forest model module loaded")