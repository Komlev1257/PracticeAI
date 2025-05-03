import os
import uuid
import cv2
import datetime
from fpdf import FPDF
from flask import make_response
from moviepy import VideoFileClip
from inference_sdk import InferenceConfiguration

custom_configuration = InferenceConfiguration(confidence_threshold=0.2)

def process_image(path, result_folder, client, model_id):
    with client.use_configuration(custom_configuration):
        response = client.infer(path, model_id=model_id)
    predictions = response.get("predictions", [])
    image = cv2.imread(path)
    for pred in predictions:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, pred['class'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    output_name = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(result_folder, output_name)
    cv2.imwrite(output_path, image)
    return {
        'type': 'image',
        'file': output_name,
        'classes': sorted(set(p['class'] for p in predictions)),
        'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def process_video(path, result_folder, client, model_id):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24

    # 1. Временный AVI-файл
    temp_name = f"{uuid.uuid4()}.avi"
    temp_path = os.path.join(result_folder, temp_name)
    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    classes = set()
    frame_written = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with client.use_configuration(custom_configuration):
            result = client.infer(frame, model_id=model_id)
        for pred in result.get("predictions", []):
            x, y, w_, h_ = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(x - w_/2), int(y - h_/2)
            x2, y2 = int(x + w_/2), int(y + h_/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, pred['class'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            classes.add(pred['class'])

        out.write(frame)
        frame_written = True

    cap.release()
    out.release()

    if not frame_written:
        os.remove(temp_path)
        raise ValueError("Видео не содержит кадров для обработки.")

    # 2. Перекодируем в mp4 через moviepy
    final_name = f"{uuid.uuid4()}.mp4"
    final_path = os.path.join(result_folder, final_name)
    clip = VideoFileClip(temp_path)
    clip.write_videofile(final_path, codec="libx264", audio=False, logger=None)
    clip.close()

    # 3. Удаляем временный .avi
    os.remove(temp_path)

    return {
        'type': 'video',
        'file': final_name,
        'classes': sorted(classes),
        'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

class PDF(FPDF):
    def header(self):
        pass
    def footer(self):
        pass

def generate_pdf_report(request_id, file_type, filename, classes, timestamp):
    pdf = PDF()
    pdf.add_page()

    font_path = os.path.join("static", "fonts", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 14)

    pdf.cell(200, 10, txt="Отчёт по обнаружению животных", ln=True, align='C')
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, f"ID запроса: {request_id}", ln=True)
    pdf.cell(0, 10, f"Тип файла: {'Видео' if file_type == 'video' else 'Изображение'}", ln=True)
    pdf.cell(0, 10, f"Дата: {timestamp}", ln=True)

    if file_type == 'image':
        img_path = os.path.join("static", "results", filename)
        pdf.image(img_path, w=150)
    else:
        pdf.cell(0, 10, f"Имя видео: {filename}", ln=True)

    pdf.cell(0, 10, "Обнаруженные объекты:", ln=True)
    for cls in classes:
        pdf.cell(0, 10, f" - {cls}", ln=True)

    # 🔥 Универсальная обработка: str или bytearray
    output_data = pdf.output(dest='S')
    if isinstance(output_data, str):
        pdf_bytes = output_data.encode('latin-1')
    else:
        pdf_bytes = bytes(output_data)

    response = make_response(pdf_bytes)
    response.headers.set('Content-Disposition', f'attachment; filename=report_{request_id}.pdf')
    response.headers.set('Content-Type', 'application/pdf')
    return response
