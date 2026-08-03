from flask import Blueprint, render_template, request, flash, redirect, url_for

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/features')
def features():
    return render_template('features.html')

@main_bp.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@main_bp.route('/faq')
def faq():
    return render_template('faq.html')

@main_bp.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name', '')
    email = request.form.get('email', '')
    message = request.form.get('message', '')
    
    if name and email and message:
        flash("Thank you for your message! Our team will respond shortly.", "success")
    else:
        flash("Please complete all fields in the contact form.", "warning")
        
    return redirect(url_for('main.index') + "#contact")
