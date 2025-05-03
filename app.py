import os
import uuid
from flask import Flask, request, render_template, session, send_from_directory
from werkzeug.utils import secure_filename
from utils import process_image, process_video, generate_pdf_report, allowed_file
from database import init_db, save_to_history, get_session_history, get_history_item
from inference_client import client, MODEL_ID
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

init_db()

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return render_template('index.html', error="Выберите допустимый файл.", history=get_session_history())

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        result_data = process_video(filepath, RESULT_FOLDER, client, MODEL_ID) if ext in {'mp4', 'avi', 'mov', 'mkv'} else process_image(filepath, RESULT_FOLDER, client, MODEL_ID)

        save_to_history(session['user_id'], result_data['type'], result_data['file'], result_data['classes'])
        result_data['id'] = get_session_history()[0]['id']
        return render_template('index.html', result=result_data, history=get_session_history())

    return render_template('index.html', history=get_session_history())

@app.route('/report/<int:request_id>')
def generate_report(request_id):
    user_id = session.get('user_id')
    row = get_history_item(request_id, user_id)
    if not row:
        return "Отчёт не найден.", 404
    return generate_pdf_report(request_id, row['file_type'], row['output_file'], row['classes'], row['timestamp'])

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory('static/results', filename, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)