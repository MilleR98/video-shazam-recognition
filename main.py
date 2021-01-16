from flask import Flask

from cfg.globals import LOG_CONF_FILE, BASE_DIR
from logger_wrapper import Log

Log.configure(path_to_config=LOG_CONF_FILE, root_dir_path=BASE_DIR)

app = Flask(__name__)


@app.route('/ping')
def hello_world():
    return 'Pong:)'


if __name__ == "__main__":
    app.run()
