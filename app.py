from flask import Flask, request, jsonify, render_template
import threading, time, datetime, signal, sys, io, base64, os
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)

# ---- Scheduler ----
last_check = {"timestamp": None, "second": None, "divisible_by_2": None}
stop_event = threading.Event()

def scheduler_loop():
    while not stop_event.is_set():
        now = datetime.datetime.now()
        sec = now.second
        divisible = (sec % 2 == 0)
        last_check.update({
            "timestamp": now.isoformat(sep=' '),
            "second": sec,
            "divisible_by_2": divisible
        })
        print(f"[{last_check['timestamp']}] seconds={sec} -> divisible_by_2={divisible}")
        time.sleep(5)

@app.route('/get_current_time')
def get_current_time():
    now = datetime.datetime.now()
    return jsonify({"current_time": now.isoformat(sep=' '), "timestamp": int(now.timestamp())})

@app.route('/last_check')
def get_last_check():
    return jsonify(last_check)

# ---- YOLO model ----
yolo_model = YOLO("yolov8n.pt")   # dùng model nhỏ YOLOv8n (nhẹ, nhanh)

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    img_bytes = img_file.read()

    # Load ảnh bằng PIL và chuyển thành numpy array
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_array = np.array(image)

    # chạy YOLO detect
    results = yolo_model.predict(source=img_array, save=False, conf=0.25, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]}
            })

    # Encode ảnh lại base64 để trả về frontend
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({"detections": detections, "image_base64": base64_img})

@app.route('/upload')
def upload_page():
    return render_template("upload.html")

# ---- Helper ----
def start_scheduler():
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    return t

def handle_sigterm(signum, frame):
    stop_event.set()
    time.sleep(0.2)
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    start_scheduler()
    app.run(host='0.0.0.0', port=8080, threaded=True)
