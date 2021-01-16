#!/bin/bash
PORT=5000
HOST='0.0.0.0'

date

export FLASK_APP=main.py

flask run --host=${HOST} --port=${PORT}