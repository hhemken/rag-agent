#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from typing import Optional, Union, Dict
import requests
import time
import re


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.api_generate = f"{host}/api/generate"
        self.api_list = f"{host}/api/list"

    def is_running(self) -> bool:
        try:
            response = requests.get(self.api_list)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def start_ollama(self) -> bool:
        try:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            # Wait for Ollama to start
            for _ in range(10):
                if self.is_running():
                    return True
                time.sleep(1)
            return False
        except FileNotFoundError:
            return False

    def ensure_model_exists(self, model: str) -> bool:
        response = requests.get(self.api_list)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if not any(m['name'] == model for m in models):
                print(f"Model {model} not found. Pulling from repository...")
                result = subprocess.run(["ollama", "pull", model],
                                        capture_output=True,
                                        text=True)
                return result.returncode == 0
            return True
        return False

    def generate(self,
                 model: str,
                 prompt: str,
                 system: Optional[str] = None) -> Optional[str]:
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            data["system"] = system

        try:
            response = requests.post(self.api_generate, json=data)
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None


def extract_code(response: str) -> Optional[Dict[str, str]]:
    """Extract code blocks from the response."""
    # Look for code blocks with language specification
    code_blocks = re.findall(r'```(\w+)\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return {lang: code.strip() for lang, code in code_blocks}

    # Look for any code blocks
    code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
    if code_blocks:
        return {"unknown": code_blocks[0].strip()}

    return None


def execute_code(code: str, language: str) -> Optional[str]:
    """Execute code if it's Python."""
    if language.lower() != 'python':
        print(f"Execution not supported for {language}")
        return None

    try:
        # Create a temporary file
        tmp_file = "temp_execution.py"
        with open(tmp_file, 'w') as f:
            f.write(code)

        # Execute the code
        result = subprocess.run([sys.executable, tmp_file],
                                capture_output=True,
                                text=True)

        # Clean up
        os.remove(tmp_file)

        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Execution error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error during execution: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Ollama CLI Interface')
    parser.add_argument('--model', required=True, help='Model to use')
    parser.add_argument('--query', help='Query string')
    parser.add_argument('--file', help='Input file containing query')
    parser.add_argument('--system', help='System prompt')
    parser.add_argument('--execute', action='store_true',
                        help='Execute code if present in response')
    parser.add_argument('--json', action='store_true',
                        help='Request JSON output')
    args = parser.parse_args()

    if not args.query and not args.file:
        parser.error("Either --query or --file must be provided")

    # Initialize client
    client = OllamaClient()

    # Check if Ollama is running, start if not
    if not client.is_running():
        print("Ollama is not running. Attempting to start...")
        if not client.start_ollama():
            print("Error: Could not start Ollama. Is it installed?")
            sys.exit(1)

    # Ensure model exists
    if not client.ensure_model_exists(args.model):
        print(f"Error: Could not load model {args.model}")
        sys.exit(1)

    # Get query from file or command line
    query = args.query
    if args.file:
        try:
            with open(args.file, 'r') as f:
                query = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

    # Add JSON format request if specified
    if args.json:
        query = f"Please provide your response in valid JSON format. {query}"

    # Generate response
    response = client.generate(args.model, query, args.system)

    if not response:
        print("Error: No response received")
        sys.exit(1)

    # Try to parse as JSON if requested
    if args.json:
        try:
            parsed = json.loads(response)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print("Warning: Response is not valid JSON")
            print(response)
    else:
        print(response)

    # Extract and optionally execute code
    code_blocks = extract_code(response)
    if code_blocks:
        print("\nExtracted code blocks:")
        for lang, code in code_blocks.items():
            print(f"\nLanguage: {lang}")
            print(code)

            if args.execute and lang.lower() == 'python':
                print("\nExecuting Python code:")
                output = execute_code(code, lang)
                if output:
                    print("Output:")
                    print(output)


if __name__ == "__main__":
    main()
