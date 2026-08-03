import sys
from app.ml.trainer import ModelTrainer

def main():
    print("========================================================")
    print("              WriteLens AI — Model Trainer               ")
    print("========================================================")
    
    trainer = ModelTrainer()
    metrics = trainer.run_training_pipeline()
    
    print("\nTraining summary:")
    print(f"Best Model Selected: {metrics['best_model_name']}")
    for model_name, res in metrics['all_models'].items():
        print(f" - {model_name:20s}: Accuracy = {res['accuracy']*100:.2f}%, F1 = {res['f1_score']*100:.2f}%, ROC AUC = {res['roc_auc']*100:.2f}%")
        
    print("\nModel training complete! All artifacts saved to trained_models/")

if __name__ == '__main__':
    main()
