from flask import Flask, render_template, Response, request
import cv2
from utils.tryon import process_frame

app = Flask(__name__)

camera = cv2.VideoCapture(0)
selected_garment = "shirt.png"


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/set_garment/<name>')
def set_garment(name):
    global selected_garment
    selected_garment = name
    return ("",204)


def generate_frames():
    global selected_garment

    while True:
        success, frame = camera.read()

        if not success:
            continue

        frame = process_frame(frame, selected_garment)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)