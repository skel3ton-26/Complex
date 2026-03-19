from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from duckduckgo_search import DDGS
import requests
import subprocess
import tempfile
import os

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/generate"
PIPER_MODEL = os.path.expanduser("~/piper_voices/en_US-amy-medium.onnx")

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return "\n".join([r['body'] for r in results])
    except:
        return ""

def ask_mistral(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }, timeout=120)
    return response.json()['response']

def text_to_speech(text):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    subprocess.run([
        "piper",
        "--model", PIPER_MODEL,
        "--output_file", tmp.name
    ], input=text.encode(), check=True)
    return tmp.name

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    use_search = data.get('search', False)

    if use_search:
        context = search_web(message)
        prompt = f"Web search results:\n{context}\n\nAnswer this question: {message}"
    else:
        prompt = message

    reply = ask_mistral(prompt)
    return jsonify({'reply': reply})

@app.route('/speak', methods=['POST'])
def speak():
    text = request.json.get('text', '')
    wav_file = text_to_speech(text)
    return send_file(wav_file, mimetype='audio/wav')

@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
