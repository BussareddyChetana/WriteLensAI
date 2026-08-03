import os
import json
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from config import Config
from app.nlp.preprocessor import TextPreprocessor
from app.nlp.stylometrics import StylometricAnalyzer

class ContentClassifier:
    def __init__(self):
        self.model_dir = Config.TRAINED_MODEL_DIR
        self.model_path = os.path.join(self.model_dir, 'model.joblib')
        self.vectorizer_path = os.path.join(self.model_dir, 'vectorizer.joblib')
        self.scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        self.metrics_path = os.path.join(self.model_dir, 'metrics.json')
        
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.metrics = {}
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.scaler = joblib.load(self.scaler_path)
                if os.path.exists(self.metrics_path):
                    with open(self.metrics_path, 'r') as f:
                        self.metrics = json.load(f)
                return True
        except Exception as e:
            print(f"Notice: Could not load trained joblib model ({e}). Using rule-assisted fallback mode.")
        return False

    def predict(self, text: str) -> dict:
        """
        Returns predictions:
        {
          "ai_probability": float (0.0 - 100.0),
          "human_probability": float (0.0 - 100.0),
          "prediction": "AI Generated" or "Human Written",
          "confidence_score": float (0.0 - 100.0)
        }
        """
        if not text or not text.strip():
            return {
                "ai_probability": 0.0,
                "human_probability": 100.0,
                "prediction": "Human Written",
                "confidence_score": 100.0
            }

        # If trained model is available
        if self.model and self.vectorizer and self.scaler:
            try:
                cleaned_text = TextPreprocessor.preprocess_for_ml(text)
                tfidf_feat = self.vectorizer.transform([cleaned_text])
                style_feat = np.array([StylometricAnalyzer.get_feature_vector(text)])
                style_scaled = self.scaler.transform(style_feat)
                
                X = hstack([tfidf_feat, csr_matrix(style_scaled)])
                
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)[0]
                    ai_prob = float(probs[1]) * 100.0
                    human_prob = float(probs[0]) * 100.0
                else:
                    pred = self.model.predict(X)[0]
                    ai_prob = 90.0 if pred == 1 else 10.0
                    human_prob = 100.0 - ai_prob
            except Exception as e:
                print(f"Inference error with joblib model: {e}. Falling back to stylometrics.")
                ai_prob, human_prob = self._stylometric_heuristic(text)
        else:
            ai_prob, human_prob = self._stylometric_heuristic(text)

        # Very short inputs (a few words) carry almost no stylistic signal,
        # yet a sparse TF-IDF vector can still make the classifier extremely
        # (and wrongly) overconfident in one direction. Pull the estimate
        # toward the uninformative 50/50 midpoint in proportion to how short
        # the text is, rather than reporting a fake high-confidence verdict.
        word_count = len(text.split())
        min_reliable_words = 20
        if word_count < min_reliable_words:
            reliability = max(0.15, word_count / min_reliable_words)
            ai_prob = 50.0 + (ai_prob - 50.0) * reliability
            human_prob = 100.0 - ai_prob

        # Cap displayed probability so the tool never claims near-certainty.
        # A synthetic-data-trained classifier that reports 99.9% is
        # overstating what it actually knows - no single stylometric/TF-IDF
        # model deserves that much trust on real-world text. Clamping to a
        # [7, 93] range keeps predictions decisive without implying
        # false certainty, regardless of how extreme the raw model output is.
        MAX_DISPLAYED_CONFIDENCE = 93.0
        MIN_DISPLAYED_CONFIDENCE = 100.0 - MAX_DISPLAYED_CONFIDENCE
        ai_prob = min(MAX_DISPLAYED_CONFIDENCE, max(MIN_DISPLAYED_CONFIDENCE, ai_prob))
        human_prob = 100.0 - ai_prob

        # Confidence reflects how far the prediction sits from the 50/50
        # decision boundary. Do NOT floor this artificially - doing so
        # reported a fake "at least 65%" confidence even on genuinely
        # borderline/uncertain predictions, which misrepresented the model.
        confidence_score = round(abs(ai_prob - human_prob), 2)

        # When the two classes are nearly tied, calling it "Likely AI
        # Generated" or "Likely Human Written" overstates what the model
        # actually knows - a 53%/47% split is not a confident verdict in
        # either direction. Report "Uncertain" instead of forcing a side.
        UNCERTAIN_BAND = 15.0  # confidence below this -> too close to call
        if confidence_score < UNCERTAIN_BAND:
            prediction = "Uncertain"
        else:
            prediction = "AI Generated" if ai_prob >= 50.0 else "Human Written"

        result = {
            "ai_probability": round(ai_prob, 2),
            "human_probability": round(human_prob, 2),
            "prediction": prediction,
            "confidence_score": round(confidence_score, 2)
        }
        if word_count < min_reliable_words:
            result["low_reliability"] = True
            result["reliability_note"] = (
                f"Text is very short ({word_count} words). Predictions on short "
                "text are far less reliable - treat this result as a rough signal, not a verdict."
            )
        return result

    def _stylometric_heuristic(self, text: str) -> tuple:
        """
        Rule-assisted calculation based on sentence burstiness, AI transition density, and TTR.
        Used as a high-precision fallback or hybrid blend.
        """
        stats = StylometricAnalyzer.extract_features(text)
        
        score = 30.0  # base baseline
        
        # 1. Low sentence variance (AI tends to produce uniform sentence lengths)
        if stats["sentence_length_std"] < 4.0:
            score += 25.0
        elif stats["sentence_length_std"] > 10.0:
            score -= 15.0

        # 2. High AI transition word density
        if stats["ai_transition_density"] > 0.15:
            score += 25.0
        elif stats["ai_transition_count"] > 0:
            score += 15.0

        # 3. Type-Token Ratio / Lexical Diversity (AI is very balanced, TTR usually around 0.5 - 0.7)
        if 0.50 <= stats["type_token_ratio"] <= 0.75 and stats["word_count"] > 30:
            score += 15.0

        # 4. Sentence length average (AI averages around 15-25 words per sentence)
        if 16 <= stats["avg_sentence_length"] <= 26:
            score += 10.0

        ai_prob = min(98.0, max(2.0, score))
        human_prob = 100.0 - ai_prob
        return ai_prob, human_prob
