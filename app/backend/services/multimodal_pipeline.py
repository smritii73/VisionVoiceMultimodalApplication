import os
from app.backend.utils.model_manager import ModelManager

class MultimodalCorrectionPipeline:
    """
    Multimodal pipeline that performs:
    1. Voice-to-text transcription (Whisper)
    2. Image + text analysis (Gemini)
    3. Correction verification (Gemini second pass)
    4. Optional TTS output (Kokoro)
    """

    def __init__(self, upload_folder="app/uploads"):
        self.manager = ModelManager(upload_folder=upload_folder)

    def run(self, image_path=None, audio_path=None, query_text=None, enable_tts=False):
        results = {}

        # Step 1: Audio → Text
        if audio_path:
            transcribed_text = self.manager.transcribe_audio(audio_path)
            results["transcribed_text"] = transcribed_text
        else:
            transcribed_text = query_text or ""

        # Step 2: Initial image + text analysis
        if image_path:
            combined_query = f"Analyze this image based on: {transcribed_text}"
            initial_response = self.manager.process_image_and_query(image_path, combined_query)
        else:
            initial_response = f"No image provided. Analyzing text only: {transcribed_text}"

        results["initial_analysis"] = initial_response

        # Step 3: Correction / Verification (Gemini re-check)
        correction_prompt = f"""
        You are a verifier model. Review the following AI-generated response for factual accuracy and clarity.
        Return a clear, corrected final answer only — no commentary or justification.
        
        Response:
        {initial_response}
        """
        corrected_response = self.manager.gemini_model.generate_content(correction_prompt).text
        results["corrected_analysis"] = corrected_response

        # Step 4: Optional TTS (with summarization safeguard)
        if enable_tts:
            # If text too long, summarize it before generating speech
            if len(corrected_response) > 800:
                summary_prompt = f"Summarize this text for audio output, keeping key points:\n{corrected_response}"
                corrected_response = self.manager.gemini_model.generate_content(summary_prompt).text
                results["summary_used"] = True  # flag for debugging or display
            else:
                results["summary_used"] = False

            audio_file = self.manager.text_to_speech(corrected_response)
            results["tts_audio"] = audio_file

        return results
