from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models.database import db, AnalysisRecord
from app.services.detector import AnalysisEngine
from app.services.file_parser import FileParser
from app.utils.helpers import generate_preview, allowed_file

analysis_bp = Blueprint('analysis', __name__, url_prefix='/analysis')

@analysis_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    if request.method == 'POST':
        input_type = request.form.get('input_type', 'text')
        text_content = ""
        file_name = None
        file_type = "TEXT"
        
        if input_type == 'file' and 'file_upload' in request.files:
            file = request.files['file_upload']
            if file and file.filename:
                if not allowed_file(file.filename):
                    flash("Invalid file format. Please upload .txt, .pdf, or .docx files.", "danger")
                    return render_template('dashboard/analyze.html')
                try:
                    text_content, file_name, file_type = FileParser.extract_text(file)
                except Exception as e:
                    flash(f"Failed to extract document text: {str(e)}", "danger")
                    return render_template('dashboard/analyze.html')
            else:
                flash("No file was uploaded.", "warning")
                return render_template('dashboard/analyze.html')
        else:
            text_content = request.form.get('pasted_text', '').strip()

        if not text_content or len(text_content.split()) < 5:
            flash("Please enter or upload at least 5 words for accurate AI analysis.", "warning")
            return render_template('dashboard/analyze.html')

        try:
            analysis_res = AnalysisEngine.analyze_text(text_content)
        except Exception as e:
            flash(f"Error during AI analysis: {str(e)}", "danger")
            return render_template('dashboard/analyze.html')

        # Title creation
        custom_title = request.form.get('title', '').strip()
        if not custom_title:
            custom_title = file_name if file_name else f"Analysis_{generate_preview(text_content, 25)}"

        # Save to Database
        record = AnalysisRecord(
            user_id=current_user.id,
            title=custom_title,
            full_text=text_content,
            text_preview=generate_preview(text_content, 250),
            file_name=file_name,
            file_type=file_type,
            prediction=analysis_res["prediction"],
            confidence_score=analysis_res["confidence_score"],
            human_prob=analysis_res["human_probability"],
            ai_prob=analysis_res["ai_probability"]
        )
        
        record.set_sentence_analysis(analysis_res["sentence_analysis"])
        record.set_explanations(analysis_res["explanations"])
        record.set_ai_sources(analysis_res["ai_sources"])
        record.set_stats(analysis_res["stats"])

        db.session.add(record)
        db.session.commit()

        flash("Analysis completed successfully!", "success")
        return redirect(url_for('analysis.result', record_id=record.id))

    return render_template('dashboard/analyze.html')

@analysis_bp.route('/result/<int:record_id>')
@login_required
def result(record_id):
    record = AnalysisRecord.query.get_or_404(record_id)
    
    # Security check: make sure user owns this record or is admin
    if record.user_id != current_user.id and not current_user.is_admin:
        flash("Unauthorized access to requested analysis record.", "danger")
        return redirect(url_for('dashboard.index'))

    return render_template('dashboard/result.html', record=record)

@analysis_bp.route('/history')
@login_required
def history():
    search_query = request.args.get('q', '').strip()
    query = AnalysisRecord.query.filter_by(user_id=current_user.id)
    
    if search_query:
        query = query.filter(
            (AnalysisRecord.title.ilike(f'%{search_query}%')) |
            (AnalysisRecord.prediction.ilike(f'%{search_query}%')) |
            (AnalysisRecord.full_text.ilike(f'%{search_query}%'))
        )
        
    records = query.order_by(AnalysisRecord.created_at.desc()).all()
    return render_template('dashboard/history.html', records=records, search_query=search_query)

@analysis_bp.route('/delete/<int:record_id>', methods=['POST'])
@login_required
def delete(record_id):
    record = AnalysisRecord.query.get_or_404(record_id)
    if record.user_id != current_user.id and not current_user.is_admin:
        flash("Unauthorized deletion request.", "danger")
        return redirect(url_for('analysis.history'))
        
    db.session.delete(record)
    db.session.commit()
    flash("Analysis record deleted successfully.", "info")
    return redirect(url_for('analysis.history'))
