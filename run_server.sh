#!/bin/bash
PORT=5000
HOST='0.0.0.0'

date

export FLASK_APP=main.py
export FLASK_ENV=development

if [[ $1 == prod ]]; then
  export FLASK_ENV=production
else
  export FLASK_ENV=development
fi

flask run --host=${HOST} --port=${PORT}
