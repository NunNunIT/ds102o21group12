import os
import joblib
import json
import py_vncorenlp
from sklearn.pipeline import Pipeline
import numpy as np

model_folder = r'E:/ASTUDY/HK6/SML/DOAN/ds102o21group12/Model_ENSEMBLEE'
vncorenlp_path = os.path.join(model_folder, r'E:/ASTUDY/HK6/SML/DOAN/ds102o21group12/Model_ENSEMBLE/VnCoreNLP')
multi_output_ensemble_path = os.path.join(model_folder, 'E:/ASTUDY/HK6/SML/DOAN/ds102o21group12/Model_ENSEMBLE/multi_output_ensemble.pkl')
tfidf_vectorizer_path = os.path.join(model_folder, 'E:/ASTUDY/HK6/SML/DOAN/ds102o21group12/Model_ENSEMBLE/tfidf_vectorizer.pkl')

def initialize_vncorenlp(vncorenlp_path):
    vncorenlp = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path, annotators=["wseg"])
    return vncorenlp

def tokenize_text(text, model):
    words = model.annotate_text(text)[0]
    return ' '.join([word["wordForm"] for word in words])

class VnCoreNLPTokenizer:
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(lambda x: tokenize_text(x, self.model))

def create_pipeline(tfidf_vectorizer_path, multi_output_ensemble_path):
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    multi_output_ensemble = joblib.load(multi_output_ensemble_path)

    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('model', multi_output_ensemble)
    ])

    return pipeline

def load_pipeline_and_predict(text):
    vncorenlp = initialize_vncorenlp(vncorenlp_path)
    pipeline = create_pipeline(tfidf_vectorizer_path, multi_output_ensemble_path)

    tokenized_text = tokenize_text(text, vncorenlp)
    predictions = pipeline.predict([tokenized_text])

    labels = ["Quality", "Price", "Environment", "Clean", "Personal", "Other"]

    data_dict = dict(zip(labels, predictions[0].tolist()))

    return data_dict

