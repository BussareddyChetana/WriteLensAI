import os
import json
from flask import Blueprint, render_template
from flask_login import login_required
from config import Config
from app.models.database import User, AnalysisRecord
from app.utils.helpers import admin_required

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/')
@login_required
@admin_required
def index():
    users = User.query.order_by(User.created_at.desc()).all()
    total_users = len(users)
    
    total_analyses = AnalysisRecord.query.count()
    ai_count = AnalysisRecord.query.filter_by(prediction='AI Generated').count()
    human_count = AnalysisRecord.query.filter_by(prediction='Human Written').count()
    
    # Load ML Model Metrics if available
    metrics_path = os.path.join(Config.TRAINED_MODEL_DIR, 'metrics.json')
    model_metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
        except Exception:
            pass

    return render_template('admin/index.html',
                           users=users,
                           total_users=total_users,
                           total_analyses=total_analyses,
                           ai_count=ai_count,
                           human_count=human_count,
                           model_metrics=model_metrics)
