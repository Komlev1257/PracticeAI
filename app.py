import os
import uuid
import datetime
import sqlite3
from flask import Flask, request, render_template, session, redirect, url_for, make_response, send_from_directory
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
from utils import process_image, process_video, generate_pdf_report

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi', 'mov', 'mkv'}

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=""
)
MODEL_ID = "stray-animal-detection/14"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    with sqlite3.connect('history.db') as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            file_type TEXT,
            output_file TEXT,
            classes TEXT,
            timestamp TEXT
        )""")
init_db()

def save_to_history(user_id, file_type, output_file, classes_list):
    with sqlite3.connect('history.db') as conn:
        conn.execute("INSERT INTO history (user_id, file_type, output_file, classes, timestamp) VALUES (?, ?, ?, ?, ?)",
                     (user_id, file_type, output_file, ", ".join(classes_list), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

def get_session_history():
    if 'user_id' not in session:
        return []
    with sqlite3.connect('history.db') as conn:
        cur = conn.execute("SELECT id, file_type, output_file, classes, timestamp FROM history WHERE user_id=? ORDER BY id DESC",
                           (session['user_id'],))
        return [{
            'id': row[0],
            'type': row[1],
            'file': row[2],
            'classes': row[3].split(", "),
            'time': row[4]
        } for row in cur.fetchall()]

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

        if ext in {'mp4', 'avi', 'mov', 'mkv'}:
            result_data = process_video(filepath, RESULT_FOLDER, CLIENT, MODEL_ID)
        else:
            result_data = process_image(filepath, RESULT_FOLDER, CLIENT, MODEL_ID)

        save_to_history(session['user_id'], result_data['type'], result_data['file'], result_data['classes'])
        result_data['id'] = get_session_history()[0]['id']
        return render_template('index.html', result=result_data, history=get_session_history())

    return render_template('index.html', history=get_session_history())

@app.route('/report/<int:request_id>')
def generate_report(request_id):
    user_id = session.get('user_id')
    with sqlite3.connect('history.db') as conn:
        cur = conn.execute("SELECT file_type, output_file, classes, timestamp FROM history WHERE id=? AND user_id=?",
                           (request_id, user_id))
        row = cur.fetchone()
        if not row:
            return "Отчёт не найден.", 404

    return generate_pdf_report(request_id, row[0], row[1], row[2].split(", "), row[3])

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory('static/results', filename, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)