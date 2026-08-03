import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    analyses = db.relationship('AnalysisRecord', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class AnalysisRecord(db.Model):
    __tablename__ = 'analysis_records'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    
    title = db.Column(db.String(150), nullable=False, default="Untitled Analysis")
    full_text = db.Column(db.Text, nullable=False)
    text_preview = db.Column(db.String(300), nullable=False)
    file_name = db.Column(db.String(150), nullable=True)
    file_type = db.Column(db.String(20), nullable=False, default="TEXT")
    
    prediction = db.Column(db.String(50), nullable=False)  # "AI Generated" or "Human Written"
    confidence_score = db.Column(db.Float, nullable=False)
    human_prob = db.Column(db.Float, nullable=False)
    ai_prob = db.Column(db.Float, nullable=False)
    
    sentence_analysis_json = db.Column(db.Text, nullable=False, default="[]")
    explanations_json = db.Column(db.Text, nullable=False, default="[]")
    ai_sources_json = db.Column(db.Text, nullable=False, default="{}")
    stats_json = db.Column(db.Text, nullable=False, default="{}")
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Helper getters/setters for JSON fields
    def set_sentence_analysis(self, data):
        self.sentence_analysis_json = json.dumps(data)

    def get_sentence_analysis(self):
        return json.loads(self.sentence_analysis_json) if self.sentence_analysis_json else []

    def set_explanations(self, data):
        self.explanations_json = json.dumps(data)

    def get_explanations(self):
        return json.loads(self.explanations_json) if self.explanations_json else []

    def set_ai_sources(self, data):
        self.ai_sources_json = json.dumps(data)

    def get_ai_sources(self):
        return json.loads(self.ai_sources_json) if self.ai_sources_json else {}

    def set_stats(self, data):
        self.stats_json = json.dumps(data)

    def get_stats(self):
        return json.loads(self.stats_json) if self.stats_json else {}

    def __repr__(self):
        return f'<AnalysisRecord {self.id} - {self.prediction}>'

class SystemStat(db.Model):
    __tablename__ = 'system_stats'

    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), unique=True, nullable=False)
    metric_value = db.Column(db.String(255), nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
