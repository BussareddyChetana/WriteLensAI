import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'writelens-ai-secret-key-production-2026-secure'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'writelens.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max limit
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'instance', 'uploads')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
    
    # Paths for ML models & Dataset
    TRAINED_MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    
    # App Settings
    APP_NAME = "WriteLens AI"
    APP_TAGLINE = "Know Who Wrote Your Words."
