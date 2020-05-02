from flask import Flask, request, jsonify
from inference import get_class_name

app = Flask(__name__)


@app.route('/', methods=['POST'])
def cnn_classifier():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        class_num, class_name = get_class_name(image_bytes=image)
        return jsonify({'class_id': class_num, 'class_name': class_name})


if __name__ == '__main__':
    app.run(debug=False)
