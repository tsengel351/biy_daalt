# BERT Embedding + Classification Model Харьцуулалт

## 🎯 Зорилго
BERT-ийн хувилбаруудаар (BERT-base, BERT-large, DistilBERT, RoBERTa, ALBERT) текстийг вектор болгож, ангилах загваруудтай (Logistic Regression, AdaBoost, Random Forest, LSTM) хослуулан IMDB мэдрэмжийн ангиллын үр дүнг харьцуулах.

## 📊 Датасет
- IMDB 50K (Train 40k / Test 10k)  
- Эерэг / Сөрөг кино шүүмж

## 🔧 Технологи
- Embedding: HuggingFace Transformers (fine-tune хийхгүй)
- ML: scikit-learn (LR, AdaBoost, RF)
- DL: TensorFlow/Keras (LSTM)
- Evaluation: RepeatedStratifiedKFold (5 folds × 4 repeats = 20 runs)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

## 🚀 Ашиглах заавар
```bash
pip install -r requirements.txt
# Embedding файлуудыг embeddings/ дотор байрлуулна
# нэршил: {model_name}_train_embeddings.npy, {model_name}_train_labels.npy, {model_name}_test_embeddings.npy, {model_name}_test_labels.npy
python main.py# biy_daalt
