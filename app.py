import os
import json
import pickle
import nltk
import numpy as np
from UseModel import response_from_chatbot
from flask import Flask, request, jsonify
from flask_cors import CORS
from json import JSONEncoder
from googletrans import Translator

nltk.download('punkt')

# App initalization
app = Flask(__name__)
CORS(app)

# Translator
translator = Translator()

# Initial GET route
@app.route('/', methods=['GET'])
def index():
    # Get parameters
    dialog1 = request.args.get("dialog1")
    language = request.args.get("language")

    try:
        # Get the prediction
        dialog2 = response_from_chatbot(dialog1)

        # Language transalation and return the final output
        final_output = ""
        if(language == "English"):
            final_output = dialog2
        elif(language == "Spanish"):
            final_output =  translator.translate(dialog1, dest="es").text
        elif(language == "German"):
            final_output =  translator.translate(dialog1, dest="de").text
        elif(language == "French"):
            final_output =  translator.translate(dialog1, dest="fr").text
        elif(language == "Russian"):
            final_output =  translator.translate(dialog1, dest="ru").text
        elif(language == "Mandarin"):
            final_output =  translator.translate(dialog1, dest="zh-cn").text

        print(final_output)

        # Final response
        response = {
            "success": True,
            "dialog2": final_output
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': 'Something went wrong!'})

# Execute the app
if __name__ == "__main__":
    app.run(host="0.0.0.0")