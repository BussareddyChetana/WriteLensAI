import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("========================================================")
    print("             WriteLens AI Web Application               ")
    print("========================================================")
    print(f"Starting server on http://127.0.0.1:{port}")
    print("Default Admin Credentials: username='admin', password='admin123'")
    print("========================================================")
    app.run(host='0.0.0.0', port=port, debug=True)
