from flask import Flask, request, render_template

from image_recognition.image_identification import identification

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/identification', methods=['POST'])
def cls():
    if request.files.get('file'):
        file = request.files.get('file')
        data = file.read()
        results = identification(data)
        return render_template('result.html', **locals())
    else:
        return index()


if __name__ == '__main__':
    app.run(debug=True)
