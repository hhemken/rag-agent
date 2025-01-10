from flask import Flask, render_template, request, jsonify
import os
from populate_database import clear_database, load_documents, split_documents, add_to_chroma
from query_data import query_rag
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import shutil


app = Flask(__name__)

# Global settings that can be modified through the interface
config = {
    "CHROMA_PATH": "chroma",
    "DATA_PATH": "/path/to/docs",
    "LLM_TO_USE": "llama2:13b"
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/database')
def database():
    return render_template('database.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/api/update_config', methods=['POST'])
def update_config():
    data = request.json
    for key in ['CHROMA_PATH', 'DATA_PATH', 'LLM_TO_USE']:
        if key in data:
            config[key] = data[key]
    return jsonify({"status": "success", "config": config})


@app.route('/api/get_config')
def get_config():
    return jsonify(config)

@app.route('/api/process_database', methods=['POST'])
def process_database():
    data = request.json
    should_reset = data.get('reset', False)

    try:
        if should_reset:
            if os.path.exists(config['CHROMA_PATH']):
                shutil.rmtree(config['CHROMA_PATH'])

        # Use config['DATA_PATH'] instead of the constant
        documents = []
        for document in os.listdir(config['DATA_PATH']):
            document_path = os.path.abspath(os.path.join(config['DATA_PATH'], document))
            if document.endswith(".pdf"):
                print(f'loading: {document_path}')
                loader = PyPDFLoader(document_path)
                documents.extend(loader.load())

        chunks = split_documents(documents)

        # Pass the CHROMA_PATH from config
        db = Chroma(persist_directory=config['CHROMA_PATH'],
                    embedding_function=get_embedding_function())
        add_to_chroma(chunks)

        return jsonify({
            "status": "success",
            "message": "Database processed successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')

    try:
        response = query_rag(question)
        # Pass the full response including sources
        return jsonify({
            "status": "success",
            "response": response,
            "full_response": response  # Include this for complete output
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/list_models')
def list_models():
    # You might want to implement actual model discovery
    return jsonify([
        "llama3.2:3b",
        "mistral"
    ])


if __name__ == '__main__':
    app.run(debug=True)