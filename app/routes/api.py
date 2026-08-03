from flask import Blueprint, jsonify, send_file, request, flash
from flask_login import login_required, current_user
from app.models.database import AnalysisRecord
from app.reports.pdf_generator import ReportPDFGenerator
from app.services.detector import AnalysisEngine

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/analyze', methods=['POST'])
def api_analyze():
    """
    JSON API Endpoint for external text content detection
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    
    if not text or len(text.split()) < 5:
        return jsonify({
            "error": "Invalid input. Please provide at least 5 words in the 'text' parameter."
        }), 400

    try:
        results = AnalysisEngine.analyze_text(text)
        return jsonify({
            "status": "success",
            "data": results
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@api_bp.route('/report/<int:record_id>/pdf')
@login_required
def download_pdf_report(record_id):
    record = AnalysisRecord.query.get_or_404(record_id)
    
    if record.user_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    record_data = {
        "id": record.id,
        "title": record.title,
        "prediction": record.prediction,
        "confidence_score": record.confidence_score,
        "human_prob": record.human_prob,
        "ai_prob": record.ai_prob,
        "created_at": record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": record.get_stats(),
        "explanations": record.get_explanations(),
        "ai_sources": record.get_ai_sources(),
        "sentence_analysis": record.get_sentence_analysis()
    }

    try:
        pdf_buffer = ReportPDFGenerator.generate_pdf(record_data)
        filename = f"WriteLens_Report_{record.id}_{record.prediction.replace(' ', '_')}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500
