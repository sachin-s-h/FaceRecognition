import os
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from demo import handler
from waitress import serve
from flask import Flask, request


app = Flask(__name__)

def array_to_bytes(x):
    np_bytes = BytesIO()
    np.save(np_bytes, np.array(x), allow_pickle=True)
    return np_bytes.getvalue()

@app.route("/", methods=["GET", "POST"])
def request_api():
    now = datetime.now()
    date = now.strftime("%m:%d:%Y_%H:%M:%S.%f")
    mode = request.form.get("mode")
    if mode == 'train':
        db_path = request.form.get("db_path")
        result = handler.write_face_features(db_path)
        return result
    elif mode == 'test':
        img1_path = request.form.get("img_path")
        db_path = request.form.get("db_path")
        img1 = cv2.imread((img1_path))
        img1 = array_to_bytes(img1)
        # db_path = './dataset'
        result = handler.validate_user(date, img1, dataset=db_path)
        return result

if __name__ == "__main__":

    serve(app, host="192.168.1.12", port=8081)