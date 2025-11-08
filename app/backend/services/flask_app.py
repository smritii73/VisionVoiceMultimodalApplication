import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import dotenv_values

# Import model manager and the new correction pipeline
from app.backend.utils.model_manager import ModelManager
from app.backend.services.multimodal_pipeline import MultimodalCorrectionPipeline

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load environment variables
config = dotenv_values('.env')
app.config['SECRET_KEY'] = config.get('FLASK_SECRET_KEY', 'default-dev-key')
app.config['UPLOAD_FOLDER'] = config.get('UPLOAD_FOLDER', 'app/uploads')
app.config['MAX_CONTENT_LENGTH'] = int(config.get('MAX_CONTENT_LENGTH', 16 * 1000 * 1000))

# Initialize models and pipelines
model_manager = ModelManager(upload_folder=app.config['UPLOAD_FOLDER'])
pipeline = MultimodalCorrectionPipeline(upload_folder=app.config['UPLOAD_FOLDER'])

# ------------------------------------------------------------------------------
# 🧩 MAIN MULTIMODAL ANALYSIS ENDPOINT (with correction logic)
# ------------------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Full multimodal analysis pipeline:
    1. Converts voice to text
    2. Analyzes image + text using Gemini
    3. Corrects the analysis (sequential refinement)
    4. Returns final corrected output (and optional TTS)
    """
    try:
        image_file = request.files.get('image')
        audio_file = request.files.get('audio')
        user_text = request.form.get('text', '')
        enable_tts = request.form.get('enable_tts', 'true').lower() == 'true'

        image_path = None
        audio_path = None

        # Save files if present
        if image_file and image_file.filename:
            image_filename = secure_filename(f"{uuid.uuid4()}_{image_file.filename}")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            image_file.save(image_path)

        if audio_file and audio_file.filename:
            audio_filename = secure_filename(f"{uuid.uuid4()}_{audio_file.filename}")
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
            audio_file.save(audio_path)

        # Run the multimodal correction pipeline
        results = pipeline.run(
            image_path=image_path,
            audio_path=audio_path,
            query_text=user_text,
            enable_tts=enable_tts
        )

        return jsonify(results), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'An error occurred during multimodal analysis.',
            'details': str(e)
        }), 500

# ------------------------------------------------------------------------------
# 🎙 AUDIO TRANSCRIPTION ENDPOINT
# ------------------------------------------------------------------------------
@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe uploaded audio to text."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({'error': 'Empty audio filename'}), 400

        audio_filename = secure_filename(f"{uuid.uuid4()}_{audio_file.filename}")
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)

        transcription = model_manager.transcribe_audio(audio_path)
        return jsonify({'transcription': transcription, 'audio_path': audio_path}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Transcription failed', 'details': str(e)}), 500

# ------------------------------------------------------------------------------
# 🧠 GEMINI IMAGE + QUERY RESPONSE ENDPOINT
# ------------------------------------------------------------------------------
@app.route('/api/generate_response', methods=['POST'])
def generate_response():
    """Generate AI response for an image + query pair."""
    try:
        data = request.form
        image_path = data.get('image_path')
        query = data.get('query')

        if not image_path or not query:
            return jsonify({'error': 'Image path and query are required'}), 400

        response_text = model_manager.process_image_and_query(image_path, query)
        return jsonify({'response': response_text}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Response generation failed', 'details': str(e)}), 500

# ------------------------------------------------------------------------------
# 🔊 TEXT-TO-SPEECH ENDPOINT
# ------------------------------------------------------------------------------
@app.route('/api/text_to_speech', methods=['POST'])
def text_to_speech():
    """Convert text to audio response."""
    try:
        data = request.json
        text = data.get('text')
        voice = data.get('voice', 'af_heart')
        speed = float(data.get('speed', 1.0))
        high_perf = data.get('high_performance', False)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if high_perf:
            speed = max(speed, 1.2)

        audio_file = model_manager.text_to_speech(text, voice=voice, speed=speed)
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file)

        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            return jsonify({'error': 'Audio generation failed'}), 500

        return jsonify({'audio_response': audio_file}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'TTS failed', 'details': str(e)}), 500

# ------------------------------------------------------------------------------
# 🎧 VOICE INFORMATION ENDPOINTS
# ------------------------------------------------------------------------------
@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Return available Kokoro voices."""
    try:
        return jsonify(model_manager.get_available_voices()), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to get voices', 'details': str(e)}), 500

@app.route('/api/voices_by_language', methods=['GET'])
def get_voices_by_language():
    """Return voices grouped by language."""
    try:
        voices_data = model_manager.get_available_voices()
        voices_by_lang = model_manager.get_voices_by_language()

        response = {}
        for lang, voice_ids in voices_by_lang.items():
            response[lang] = [
                {**voices_data[v_id], 'id': v_id}
                for v_id in voice_ids if v_id in voices_data
            ]
        return jsonify(response), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to group voices', 'details': str(e)}), 500

# ------------------------------------------------------------------------------
# ⚙️ HEALTH CHECK
# ------------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple API health check."""
    return jsonify({'status': 'ok'}), 200

# ------------------------------------------------------------------------------
# 🚀 APP ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
