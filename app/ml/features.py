import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from app.nlp.preprocessor import TextPreprocessor
from app.nlp.stylometrics import StylometricAnalyzer

class HybridFeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_df=0.9
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, texts: list):
        cleaned_texts = [TextPreprocessor.preprocess_for_ml(t) for t in texts]
        tfidf_mat = self.vectorizer.fit_transform(cleaned_texts)
        
        style_feats = np.array([StylometricAnalyzer.get_feature_vector(t) for t in texts])
        style_scaled = self.scaler.fit_transform(style_feats)
        
        self.is_fitted = True
        return hstack([tfidf_mat, csr_matrix(style_scaled)])

    def transform(self, texts: list):
        cleaned_texts = [TextPreprocessor.preprocess_for_ml(t) for t in texts]
        tfidf_mat = self.vectorizer.transform(cleaned_texts)
        
        style_feats = np.array([StylometricAnalyzer.get_feature_vector(t) for t in texts])
        style_scaled = self.scaler.transform(style_feats)
        
        return hstack([tfidf_mat, csr_matrix(style_scaled)])
