from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app.models.database import db, User

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template('auth/register.html')
            
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('auth/register.html')
            
        if User.query.filter_by(username=username).first():
            flash("Username is already taken. Please choose another.", "danger")
            return render_template('auth/register.html')
            
        if User.query.filter_by(email=email).first():
            flash("Email address is already registered.", "danger")
            return render_template('auth/register.html')
            
        # First registered user can be admin
        is_first_user = User.query.count() == 0
        
        user = User(username=username, email=email, is_admin=is_first_user)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('auth.login'))
        
    return render_template('auth/register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
        
    if request.method == 'POST':
        login_input = request.form.get('login_input', '').strip()
        password = request.form.get('password', '')
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter((User.username == login_input) | (User.email == login_input.lower())).first()
        
        if not user or not user.check_password(password):
            flash("Invalid username/email or password.", "danger")
            return render_template('auth/login.html')
            
        login_user(user, remember=remember)
        flash(f"Welcome back, {user.username}!", "success")
        
        next_page = request.args.get('next')
        return redirect(next_page or url_for('dashboard.index'))
        
    return render_template('auth/login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('main.index'))

@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        new_password = request.form.get('new_password', '')
        
        existing_email_user = User.query.filter(User.email == email, User.id != current_user.id).first()
        if existing_email_user:
            flash("Email address is already in use by another account.", "danger")
        else:
            current_user.email = email
            if new_password:
                current_user.set_password(new_password)
            db.session.commit()
            flash("Profile updated successfully!", "success")
            
    # Calculate user analysis statistics
    user_records = current_user.analyses.all()
    total_analyses = len(user_records)
    ai_count = sum(1 for r in user_records if r.prediction == 'AI Generated')
    human_count = sum(1 for r in user_records if r.prediction == 'Human Written')
    
    return render_template('auth/profile.html',
                           total_analyses=total_analyses,
                           ai_count=ai_count,
                           human_count=human_count)
