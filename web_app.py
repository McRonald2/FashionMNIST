from flask import Flask, request, render_template
from inference import get_class_name

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def cnn_classifier():
    if request.method == 'GET':
        return render_template(template_name_or_list=['index.html'])
    if request.method == 'POST':
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        pred_class, cat_name = get_class_name(image_bytes=image)
        return render_template('result.html', class_idx=pred_class, class_name=cat_name)


if __name__ == '__main__':
    app.run(debug=False)
