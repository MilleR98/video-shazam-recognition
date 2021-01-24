from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from cfg.globals import LOG_CONF_FILE, BASE_DIR
from logger_wrapper import Log

Log.configure(path_to_config=LOG_CONF_FILE, root_dir_path=BASE_DIR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=None)


@app.route('/ping')
def hello_world():
    return 'Pong:)'


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
    socketio.run(app, debug=True)
