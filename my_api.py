from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import os
import my_yolov6
import cv2
import json


from flask import request

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static'

yolov6_model = my_yolov6.my_yolov6("weights/yolov6s.pt", 'cpu', 'data/coco.yaml', 640, True)

@app.route('/', methods=['POST'] )
@cross_origin(origin='*')
def predict_yolov6():
    image = request.files['file']
    if image:
        # Lưu file
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        print("Save= ", path_to_save)
        image.save(path_to_save)

        frame = cv2.imread(path_to_save )

        # Nhận diện qua model YOLOv6

        objects = yolov6_model.infer(frame)
        print(objects)
        json_object = json.dumps(objects)
        print(json_object)

        del frame
    # Trả về đường dẫn tới file ảnh đã bounding box

        return json_object
    return 'Upload file to detect: '



# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')