from flask import request
from flask import Flask
from Disposition import Disposit
import json

app = Flask(__name__)

dispoit = Disposit()


@app.route('/', methods=['GET', 'POST'])
def index():
    data = request.args.get('data')
    print(data)
    return dispoit.precidt(data)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")