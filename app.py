from flask import Flask, render_template, request, jsonify
import os
from populate_database import clear_database, load_documents, split_documents, add_to_chroma
from query_data import query_rag
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import shutil
from ollama import Client

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
        # Just return the full response without trying to split it
        return jsonify({
            "status": "success",
            "response": response,  # The full response including sources
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# @app.route('/api/list_models')
# def list_models():
#     # You might want to implement actual model discovery
#     return jsonify([
#         "llama3.2:3b",
#         "mistral"
#     ])

@app.route('/api/list_models')
def list_models():
    try:
        client = Client(host='http://localhost:11434')
        response = client.list()

        # Debug logging
        print(f"Raw Ollama response: {response}")

        model_names = []

        # Convert response to string and split into individual Model entries
        response_str = str(response)
        if response_str.startswith('models=['):
            # Remove the 'models=[' prefix and trailing ']'
            models_str = response_str[7:-1]

            # Split by '), Model(' to get individual model entries
            model_entries = models_str.split('), Model(')

            for entry in model_entries:
                # Clean up the entry
                entry = entry.replace('Model(', '').replace(')', '')

                # Find the model name
                if 'model=' in entry:
                    # Extract the text between model=' and the next comma or quote
                    model_name = entry.split('model=')[1].split(',')[0].strip("'")
                    model_names.append(model_name)

        if not model_names:
            print("No valid models found in response")
            return jsonify(["No models available"])

        print(f"Successfully extracted model names: {model_names}")
        return jsonify(model_names)

    except Exception as e:
        print(f"Error listing models: {str(e)}")
        # Include the error message in the response for debugging
        return jsonify([f"Error: {str(e)}"])

@app.route('/api/list_databases')
def list_databases():
    try:
        # Look for directories containing chroma.sqlite3
        databases = []
        for item in os.listdir():
            db_path = os.path.join(item, 'chroma.sqlite3')
            if os.path.isdir(item) and os.path.exists(db_path):
                databases.append(item)

        print(f"Found databases: {databases}")
        return jsonify(databases)
    except Exception as e:
        print(f"Error listing databases: {str(e)}")
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)
