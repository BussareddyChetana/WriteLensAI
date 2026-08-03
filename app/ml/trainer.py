import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

from config import Config
from app.nlp.preprocessor import TextPreprocessor
from app.ml.features import HybridFeatureExtractor

def ensure_dataset_exists():
    """
    Checks if dataset/ai_vs_human.csv exists; if not, generates a sample initial dataset
    so model training works out-of-the-box before Kaggle dataset placement.
    """
    os.makedirs(Config.DATASET_DIR, exist_ok=True)
    dataset_path = os.path.join(Config.DATASET_DIR, 'ai_vs_human.csv')
    
    if os.path.exists(dataset_path):
        return dataset_path
        
    print("Dataset file not found at dataset/ai_vs_human.csv. Generating sample dataset...")
    
    human_samples = [
        "I remember the summer of 2018 when my family drove up to Maine. We stayed in a little cabin near the lake, where the pine trees smelled fresh after the morning rain. My grandfather taught me how to fish with old wooden lures.",
        "Honestly, I'm not super excited about the new policy changes at work. It feels like management is adding extra paperwork without really fixing the main communication problem between teams.",
        "Cooking risotto requires patience and a good wooden spoon. You can't just dump all the broth in at once; you have to let the rice slowly absorb every ladleful while stirring continuously.",
        "The local library hosted a book swap yesterday afternoon. I ended up trading three mystery novels for a copy of a classic travel memoir, which I've already started reading over coffee.",
        "Our team ran into several unexpected bugs during the sprint test. Joe found an issue with memory leaks when loading large images, so we spent most of Thursday debugging C++ pointers.",
        "Walking through the city park in October is always my favorite routine. The leaves turn deep shade of orange and crunch under your shoes, and vendors sell hot apple cider by the fountain.",
        "I was trying to fix the squeaky hinges on the pantry door this morning. After trying WD-40 without much luck, I pulled out the hinge pins and sanded off the slight rust.",
        "My dog gets incredibly excited whenever someone opens a bag of potato chips. He can be fast asleep in another room, but the sound instantly brings him sprinting into the kitchen.",
        "Last night we decided to try making homemade pizza from scratch. The dough rose perfectly on the counter, and we topped it with fresh mozzarella, fresh basil, and cherry tomatoes from our garden.",
        "Attending the evening symphony concert was an unforgettable experience. The acoustic quality inside the grand hall brought out every warm tone of the cello section during the second movement.",
        "Repairing an old vintage bicycle requires patience and hard-to-find components. I spent three hours polishing the chrome handlebars and adjusting the gear shift cable.",
        "We spent the weekend camping near the river trail. Setting up the tent before dusk was a bit challenging because of strong winds, but sitting by the campfire under the stars made it all worth it.",
        "My morning routine usually starts with grinding fresh coffee beans. The rich aroma fills the kitchen while I review my notes and set priorities for the workday ahead.",
        "Traveling by train across the countryside offers a peaceful perspective. You can watch quiet farmland, ancient stone bridges, and small coastal towns drift past the window.",
        "Writing in a leather-bound journal before going to sleep helps clear my thoughts. It is a simple habit that keeps me grounded regardless of how chaotic the week gets."
    ]

    ai_samples = [
        "In contemporary societal discourse, artificial intelligence represents a pivotal advancement in technological capability. Furthermore, its multifaceted application fosters innovation across diverse domains.",
        "To summarize the key findings, machine learning architectures provide paramount efficiency in statistical data analysis. Consequently, understanding these foundational principles is crucial.",
        "In conclusion, renewable energy solutions serve as a vital catalyst for environmental sustainability. It is important to note that holistic approaches underscore long-term prosperity.",
        "Furthermore, the integration of automated frameworks streamlines operational workflows. It is paramount that organizations embrace continuous optimization to remain competitive in the global market.",
        "Delving into the realm of digital transformation reveals a tapestry of interconnected technological ecosystems. Therefore, a comprehensive strategy is essential for navigating modern challenges.",
        "It is crucial to recognize that quantum computing underscores a paradigm shift in data processing capabilities. Consequently, researchers continue to explore its transformative potential.",
        "In summary, effective communication serves as a cornerstone of successful project management. Moreover, leveraging structured communication channels fosters clarity and mitigates potential risks.",
        "The multifaceted nature of urban planning requires a balanced approach to economic growth and ecological preservation. Nevertheless, sustainable infrastructure remains a top priority.",
        "Advancements in natural language processing enable machines to parse complex linguistic structures with unprecedented accuracy. Thus, continuous research into model interpretability remains vital.",
        "The implementation of cloud infrastructure enhances organizational agility and data redundancy. Consequently, enterprise architectures increasingly rely on scalable microservices.",
        "In light of recent technological developments, cyber security protocols must evolve dynamically. It is imperative that system administrators maintain vigilance against emerging threats.",
        "To foster collaborative innovation, cross-functional teams must align their strategic objectives. Furthermore, transparent communication channels facilitate seamless knowledge sharing across departments.",
        "Delving into macroeconomic trends reveals the paramount importance of adaptive fiscal policies. Therefore, policymakers must navigate volatile market indicators with strategic precision.",
        "In conclusion, cognitive computing systems continue to redefine human-computer interaction paradigms. It is important to note that ethical guidelines serve as a crucial foundation.",
        "The integration of edge computing nodes minimizes latency in real-time sensor analytics. Consequently, industrial automation frameworks achieve optimal operational efficiency."
    ]

    # Create variation by combining phrases
    all_rows = []
    for h in human_samples:
        all_rows.append({'text': h, 'generated': 0})
    for a in ai_samples:
        all_rows.append({'text': a, 'generated': 1})

    for i in range(len(human_samples)):
        for j in range(i+1, min(i+4, len(human_samples))):
            all_rows.append({'text': human_samples[i] + " " + human_samples[j], 'generated': 0})
            all_rows.append({'text': ai_samples[i] + " " + ai_samples[j], 'generated': 1})

    df = pd.DataFrame(all_rows).sample(frac=1, random_state=42)
    df.to_csv(dataset_path, index=False)
    print(f"Sample dataset successfully created at {dataset_path} with {len(df)} records.")
    return dataset_path

class ModelTrainer:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path or os.path.join(Config.DATASET_DIR, 'ai_vs_human.csv')
        self.output_dir = Config.TRAINED_MODEL_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def run_training_pipeline(self):
        print("=== WriteLens AI Model Training Pipeline ===")
        
        # 1. Dataset verification / generation
        if not os.path.exists(self.dataset_path):
            self.dataset_path = ensure_dataset_exists()
            
        print(f"Loading dataset from: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        
        # Determine column names (handle 'generated' or 'label' or 'target')
        label_col = 'generated' if 'generated' in df.columns else ('label' if 'label' in df.columns else 'target')
        text_col = 'text' if 'text' in df.columns else 'content'
        
        # 2. Preprocessing: remove duplicates, missing values
        initial_len = len(df)
        df = df.dropna(subset=[text_col, label_col])
        df = df.drop_duplicates(subset=[text_col])
        print(f"Cleaned dataset: {len(df)} rows (removed {initial_len - len(df)} duplicates/nulls)")

        # If rows are tagged with a 'source' column, upweight rows that
        # aren't from the single largest bulk dataset. A single dataset
        # (e.g. one Kaggle essay corpus) can otherwise dominate training
        # purely by row count and drown out smaller, more topically/
        # stylistically diverse data that helps the model generalize to
        # real-world text outside that one domain.
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            majority_source = source_counts.idxmax()
            print(f"Dataset sources: {dict(source_counts)} (majority: {majority_source})")
            weights = df['source'].apply(lambda s: 1.0 if s == majority_source else 6.0).values
        else:
            weights = np.ones(len(df))

        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).values
        
        # 3. Train/Test Split (80/20)
        X_train_raw, X_test_raw, y_train, y_test, w_train, w_test = train_test_split(
            texts, labels, weights, test_size=0.20, random_state=42, stratify=labels
        )
        print(f"Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")
        
        # 4. Feature Extraction (TF-IDF + Stylometrics)
        print("Extracting hybrid TF-IDF & Stylometric features...")
        extractor = HybridFeatureExtractor(max_features=15000)
        X_train = extractor.fit_transform(X_train_raw)
        X_test = extractor.transform(X_test_raw)
        
        # 5. Model Candidates
        models = {
            "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        }
        
        results = {}
        fitted_models = {}
        best_name = None
        best_f1 = -1.0
        
        for name, clf in models.items():
            print(f"\nTraining {name}...")
            try:
                clf.fit(X_train, y_train, sample_weight=w_train)
            except TypeError:
                clf.fit(X_train, y_train)
            fitted_models[name] = clf
            
            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred
                
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = acc
                
            results[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4)
            }
            
            print(f"Results for {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
                
        print(f"\n>>> Best Performing Model: {best_name} (F1 Score: {best_f1:.4f}) <<<")
        best_model = fitted_models[best_name]
        
        # 6. Save Model Artifacts
        model_path = os.path.join(self.output_dir, 'model.joblib')
        vectorizer_path = os.path.join(self.output_dir, 'vectorizer.joblib')
        scaler_path = os.path.join(self.output_dir, 'scaler.joblib')
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        
        joblib.dump(best_model, model_path)
        joblib.dump(extractor.vectorizer, vectorizer_path)
        joblib.dump(extractor.scaler, scaler_path)
        
        metrics_payload = {
            "best_model_name": best_name,
            "all_models": results,
            "best_metrics": results[best_name],
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw)
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_payload, f, indent=4)
            
        # 7. Generate Visualizations (Confusion Matrix & ROC Curve)
        self.generate_plots(best_model, X_test, y_test, best_name)
        
        print("\nTraining completed successfully! Artifacts saved to:", self.output_dir)
        return metrics_payload

    def generate_plots(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # ROC Curve Plot
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='indigo', lw=2, label=f'ROC curve (area = {auc_val:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=150)
            plt.close()

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
