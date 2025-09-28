from flask import Flask, request, jsonify, Response
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

# Configuration
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
NVIDIA_BASE_URL = os.environ.get('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/v1/generate', methods=['POST'])
def generate():
    """Legacy Ooba API endpoint that proxies to NVIDIA NIM"""
    try:
        # Get the request data
        data = request.get_json()
        
        # Extract parameters from Ooba format
        prompt = data.get('prompt', '')
        max_new_tokens = data.get('max_new_tokens', 512)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        top_k = data.get('top_k', 40)
        repetition_penalty = data.get('repetition_penalty', 1.0)
        stop = data.get('stopping_strings', [])
        
        # Convert to NVIDIA NIM format
        nvidia_payload = {
            "model": "meta/llama-3.1-405b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop if stop else None,
            "stream": data.get('stream', False)
        }
        
        # Make request to NVIDIA NIM
        headers = {
            'Authorization': f'Bearer {NVIDIA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{NVIDIA_BASE_URL}/chat/completions',
            headers=headers,
            json=nvidia_payload,
            stream=nvidia_payload.get('stream', False)
        )
        
        if response.status_code != 200:
            return jsonify({
                'error': f'NVIDIA API error: {response.status_code}',
                'details': response.text
            }), response.status_code
        
        # Handle streaming response
        if nvidia_payload.get('stream', False):
            def generate_streaming_response():
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_part = line[6:]  # Remove 'data: ' prefix
                            if data_part.strip() == '[DONE]':
                                break
                            
                            try:
                                chunk_data = json.loads(data_part)
                                content = chunk_data['choices'][0]['delta'].get('content', '')
                                
                                # Convert to Ooba streaming format
                                ooba_chunk = {
                                    'event': 'text_stream',
                                    'message_num': 0,
                                    'text': content
                                }
                                yield f"data: {json.dumps(ooba_chunk)}\n\n"
                            except json.JSONDecodeError:
                                continue
                
                # Send final event
                final_event = {
                    'event': 'stream_end',
                    'message_num': 0
                }
                yield f"data: {json.dumps(final_event)}\n\n"
            
            return Response(
                generate_streaming_response(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
        
        # Handle non-streaming response
        nvidia_response = response.json()
        generated_text = nvidia_response['choices'][0]['message']['content']
        
        # Convert to Ooba format
        ooba_response = {
            'results': [{
                'text': generated_text,
                'tokens': len(generated_text.split())  # Approximate token count
            }]
        }
        
        return jsonify(ooba_response)
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/v1/model', methods=['GET'])
def get_model():
    """Return current model info in Ooba format"""
    return jsonify({
        'result': 'meta/llama-3.1-405b-instruct'
    })

@app.route('/api/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'model_names': [
            'meta/llama-3.1-405b-instruct',
            'meta/llama-3.1-70b-instruct',
            'meta/llama-3.1-8b-instruct'
        ]
    })

@app.route('/api/v1/info/version', methods=['GET'])
def version():
    """Return version info"""
    return jsonify({
        'version': '1.0.0',
        'proxy': 'nvidia-nim-ooba-proxy'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
