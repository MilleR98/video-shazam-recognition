import datetime
import os

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from cfg.globals import LOG_CONF_FILE, BASE_DIR, DATA_DIR
from logger_wrapper import Log
from processing.video_processing import VideoProcessing

Log.configure(path_to_config=LOG_CONF_FILE, root_dir_path=BASE_DIR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=None)


@app.route('/ping')
def ping():
    return 'Pong:)'


@app.route('/api/recognize', methods=['POST'])
def search_video():
    print(request)
    input_video = request.files['input_file']
    unique_timestamp = str(datetime.datetime.now().timestamp()) + '_'
    path_to_input = DATA_DIR / 'temp' / (unique_timestamp + input_video.filename)
    input_video.save(path_to_input)
    try:
        result = VideoProcessing().query_video(path_to_fragment=str(path_to_input))
    finally:
        os.remove(path_to_input)

    return jsonify(result)


@app.route('/', methods=['GET'])
def upload_file():
    return render_template('index.html', sync_mode=socketio.async_mode)


@socketio.on('connect')
def test_connect():
    emit('my_response', 'Client connected')


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


@socketio.on('events')
def handle_my_custom_event(json):
    print('received json: ' + str(json))


if __name__ == "__main__":
    socketio.run(app)
