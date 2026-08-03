from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.models.database import AnalysisRecord

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@dashboard_bp.route('/')
@login_required
def index():
    recent_analyses = AnalysisRecord.query.filter_by(user_id=current_user.id)\
                                         .order_by(AnalysisRecord.created_at.desc())\
                                         .limit(10).all()
                                         
    total_analyses = AnalysisRecord.query.filter_by(user_id=current_user.id).count()
    ai_count = AnalysisRecord.query.filter_by(user_id=current_user.id, prediction='AI Generated').count()
    human_count = AnalysisRecord.query.filter_by(user_id=current_user.id, prediction='Human Written').count()

    return render_template('dashboard/index.html',
                           recent_analyses=recent_analyses,
                           total_analyses=total_analyses,
                           ai_count=ai_count,
                           human_count=human_count)
