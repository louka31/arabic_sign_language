from flask import Flask, render_template, Response
from camera import Video

app=Flask(__name__)


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/ASLToText.html')
def index1():
    return render_template('ASLToText.html')


@app.route('/TextToASL.html')
def index2():
    return render_template('TextToASL.html')

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

@app.route('/video')

def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)