import os
import json
import pickle
import nltk
import numpy as np
from UseModel import response_from_chatbot
from flask import Flask, request, jsonify
from flask_cors import CORS
from json import JSONEncoder

# App initalization
app = Flask(__name__)
CORS(app)

# Initial GET route
@app.route('/', methods=['GET'])
def index():
    dialog1 = request.args.get("dialog1")
    print(dialog1)
    try:
        dialog2 = response_from_chatbot(dialog1)
        print(dialog2)
        response = {
            "success": True,
            "dialog2": dialog2
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': 'Something went wrong!'})

# Execute the app
if __name__ == "__main__":
    app.run(host="0.0.0.0")