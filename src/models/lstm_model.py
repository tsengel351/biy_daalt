"""
LSTM загвар (TensorFlow/Keras)
"""
import numpy as np
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


class LSTMModel: 
    """LSTM классификатор"""
    
    def __init__(self, max_words: int = 10000, max_length: int = 500):
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.history = None
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            self.tf = tf
            self.keras = keras
            self.available = True
        except ImportError: 
            print("⚠️  TensorFlow not installed.  LSTM not available.")
            self.available = False
    
    def _build_model(self):
        """LSTM загвар бүтээх"""
        if not self.available:
            raise RuntimeError("TensorFlow not available")
        
        layers = self.keras.layers
        
        model = self.keras.Sequential([
            layers. Embedding(self.max_words, config.LSTM_CONFIG['embedding_dim'], 
                           input_length=self.max_length),
            layers.SpatialDropout1D(0.2),
            layers. LSTM(config.LSTM_CONFIG['lstm_units'], 
                       dropout=config.LSTM_CONFIG['dropout'],
                       recurrent_dropout=config.LSTM_CONFIG['recurrent_dropout'],
                       return_sequences=True),
            layers. LSTM(config.LSTM_CONFIG['lstm_units'] // 2,
                       dropout=config. LSTM_CONFIG['dropout'],
                       recurrent_dropout=config. LSTM_CONFIG['recurrent_dropout']),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy', self.keras.metrics.AUC(name='auc')])
        return model
    
    def prepare_data(self, texts: List[str], fit_tokenizer: bool = True):
        """Текстүүдийг sequence болгох"""
        if not self.available:
            raise RuntimeError("TensorFlow not available")
        
        from tensorflow. keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        if fit_tokenizer:
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded
    
    def train(self, X_train: List[str], y_train: np.ndarray,
              X_val: List[str] = None, y_val: np.ndarray = None) -> Dict[str, Any]: 
        """Загварыг сургах"""
        if not self.available:
            raise RuntimeError("TensorFlow not available")
        
        print("Training LSTM...")
        
        X_train_seq = self.prepare_data(X_train, fit_tokenizer=True)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq = self.prepare_data(X_val, fit_tokenizer=False)
            validation_data = (X_val_seq, y_val)
        
        self. model = self._build_model()
        print(self.model.summary())
        
        callbacks = [
            self.keras.callbacks.EarlyStopping(monitor='val_loss' if validation_data else 'loss',
                                              patience=3, restore_best_weights=True),
            self.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss',
                                                   factor=0.5, patience=2, min_lr=1e-7)
        ]
        
        self.history = self.model.fit(
            X_train_seq, y_train,
            batch_size=config.LSTM_CONFIG['batch_size'],
            epochs=config.LSTM_CONFIG['epochs'],
            validation_data=validation_data,
            validation_split=config.LSTM_CONFIG['validation_split'] if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'history': self.history. history,
            'final_train_loss': self.history.history['loss'][-1],
            'final_train_acc': self.history.history['accuracy'][-1]
        }
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Урьдчилсан таамаглал"""
        X_seq = self.prepare_data(X, fit_tokenizer=False)
        predictions = self.model.predict(X_seq)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X:  List[str]) -> np.ndarray:
        """Магадлалын урьдчилсан таамаглал"""
        X_seq = self. prepare_data(X, fit_tokenizer=False)
        predictions = self.model.predict(X_seq)
        return np.column_stack([1 - predictions, predictions])
    
    def save(self, model_path: str, tokenizer_path: str):
        """Загвар хадгалах"""
        if self.available:
            self.model.save(model_path)
            import pickle
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"LSTM model saved to {model_path}")
    
    def load(self, model_path: str, tokenizer_path: str):
        """Загвар ачаалах"""
        if self.available:
            self.model = self. keras.models.load_model(model_path)
            import pickle
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"LSTM model loaded from {model_path}")


if __name__ == "__main__":
    print("✓ LSTM model module loaded")