import os
import gradio as gr
import requests
from dotenv import dotenv_values

def launch_gradio():
    """Create and launch the Gradio interface for the Multimodal Correction Agent."""

    # Unified backend API
    ANALYZE_API_URL = "http://localhost:5000/analyze"
    VOICES_API_URL = "http://localhost:5000/api/voices_by_language"

    config = dotenv_values('.env')

    # --------------------------------------------------------------------------
    # 🔊 Fetch available voices from backend
    # --------------------------------------------------------------------------
    try:
        response = requests.get(VOICES_API_URL)
        if response.status_code == 200:
            voices_by_language = response.json()
            languages = list(voices_by_language.keys())
            voices_options = {
                lang: {f"{v['name']} ({v['gender']})": v['id'] for v in voices}
                for lang, voices in voices_by_language.items()
            }
        else:
            languages = ["English (African)"]
            voices_options = {"English (African)": {"Heart (Female)": "af_heart"}}
    except Exception as e:
        print(f"Error fetching voices: {str(e)}")
        languages = ["English (African)"]
        voices_options = {"English (African)": {"Heart (Female)": "af_heart"}}

    # --------------------------------------------------------------------------
    # 🎯 Unified multimodal correction function
    # --------------------------------------------------------------------------
    def analyze_multimodal_inputs(image, audio, text, language, voice_selection, speed):
        """
        Send all inputs (image, audio, text) to backend for multimodal correction.
        The backend returns:
        - initial_analysis
        - corrected_analysis
        - optional tts_audio
        """
        if image is None and audio is None and not text:
            return "Please provide at least an image, audio, or text input.", "", None

        files = {}
        if image:
            files["image"] = open(image, "rb")
        if audio:
            files["audio"] = open(audio, "rb")
        data = {
            "text": text,
            "enable_tts": "true",
        }

        try:
            response = requests.post(ANALYZE_API_URL, files=files, data=data)
            if response.status_code == 200:
                result = response.json()

                # Extract results
                initial_analysis = result.get("initial_analysis", "")
                corrected_analysis = result.get("corrected_analysis", "")

                # Handle optional TTS output
                audio_file = result.get("tts_audio")
                audio_response_path = (
                    os.path.join(config.get("UPLOAD_FOLDER", "app/uploads"), audio_file)
                    if audio_file
                    else None
                )

                return initial_analysis, corrected_analysis, audio_response_path

            else:
                return f"Error: {response.status_code} - {response.text}", "", None

        except Exception as e:
            return f"Error communicating with backend: {str(e)}", "", None

    # --------------------------------------------------------------------------
    # 🎛 UI Controls
    # --------------------------------------------------------------------------
    with gr.Blocks(title="Vision-Voice Multimodal Correction Agent", theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            "<h1 style='text-align: center;'>🎯 Vision-Voice Multimodal Correction Agent</h1>"
            "<p style='text-align: center; font-size: 1.1em;'>"
            "An AI agent that analyzes multimodal inputs (image, voice, and text) and corrects its own predictions for higher accuracy."
            "</p>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="📸 Upload Image (Optional)", type="filepath")
                audio_input = gr.Audio(
                    label="🎙 Record Voice Query (Optional)",
                    type="filepath",
                    sources=["microphone"]
                )
                text_input = gr.Textbox(label="💬 Enter Text Query (Optional)", placeholder="Or type your question here...")

                with gr.Accordion("🔊 Voice Settings for Response (Optional)", open=False):
                    language_dropdown = gr.Dropdown(
                        choices=languages,
                        value=languages[0] if languages else None,
                        label="Language"
                    )
                    voice_dropdown = gr.Dropdown(
                        choices=list(voices_options.get(languages[0], {}).keys()) if languages else [],
                        value=list(voices_options.get(languages[0], {}).keys())[0] if languages else None,
                        label="Voice"
                    )
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speech Speed"
                    )

                analyze_btn = gr.Button("🔍 Analyze & Correct", variant="primary")

            with gr.Column(scale=1):
                initial_response = gr.Textbox(label="🧩 Initial AI Analysis", lines=4, interactive=False)
                corrected_response = gr.Textbox(label="✅ Final Corrected AI Response", lines=6, interactive=False)
                response_audio = gr.Audio(label="🔈 Voice Output (Corrected Response)")

        # ----------------------------------------------------------------------
        # 🔄 Event Handlers
        # ----------------------------------------------------------------------
        def update_voice_options(language):
            """Update voice dropdown when language changes."""
            if language in voices_options:
                voices = voices_options[language]
                default_voice = list(voices.keys())[0] if voices else None
                return gr.update(choices=list(voices.keys()), value=default_voice)
            else:
                return gr.update(choices=[], value=None)

        language_dropdown.change(
            fn=update_voice_options,
            inputs=[language_dropdown],
            outputs=[voice_dropdown]
        )

        analyze_btn.click(
            fn=analyze_multimodal_inputs,
            inputs=[image_input, audio_input, text_input, language_dropdown, voice_dropdown, speed_slider],
            outputs=[initial_response, corrected_response, response_audio],
            show_progress=True
        )

        # ----------------------------------------------------------------------
        # ℹ️ How It Works Section
        # ----------------------------------------------------------------------
        with gr.Row():
            gr.Markdown("""
            ### 💡 How It Works
            1. Upload an image, record your voice, or type your question.  
            2. The system performs **multimodal reasoning** using vision, voice, and text models.  
            3. A second pass by the AI **corrects the initial response** for better precision.  
            4. The final output can also be **spoken back** using natural-sounding voices.  
            """)

    iface.launch(share=False)
    return iface


if __name__ == "__main__":
    launch_gradio()
