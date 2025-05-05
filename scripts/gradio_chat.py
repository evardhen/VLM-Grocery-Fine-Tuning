import os
import json
import gradio as gr
from utils.streamed_chat import StreamedChat  # our streamlined chat class
from llamafactory.extras.misc import torch_gc  # used to clear VRAM

# Import our TTS wrapper.
from utils.tts import KokoroTTS

# Import the ASR pipeline and numpy.
from transformers import pipeline
import numpy as np

# Create a global TTS instance (loads the model once).
tts_instance = KokoroTTS(lang_code='a')
_audio_playing = False
# Create a global ASR (speech-to-text) transcriber instance.
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


def transcribe_audio(audio):
    """
    Transcribes the provided audio and clears the audio component.
    """
    if audio is None:
        return gr.update(), gr.update(value=None)
    sr, y = audio
    # Convert to mono if needed.
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y /= max_val
    transcription = transcriber({"sampling_rate": sr, "raw": y})["text"]
    return transcription, gr.update(value=None)


def get_inference_configs(config_dir="inference_configs/fixed_inference_image_resolution_589824"):
    """Scans the specified folder for .yaml files and returns a list of filenames."""
    if not os.path.isdir(config_dir):
        return []
    return [f for f in os.listdir(config_dir) if f.endswith('.yaml')]


def get_datapoints(json_path="data/lingoQA_evaluation.json"):
    """Loads datapoints from the specified JSON file."""
    if not os.path.isfile(json_path):
        return [], []
    with open(json_path, "r") as f:
        data = json.load(f)
    choices = [f"Datapoint {i} (ID: {dp['question_id']})" for i, dp in enumerate(data)]
    return data, choices


def clear_images():
    return [], []


def create_chat_ui():
    """
    Creates a UI for a streamed chat interface, including model and datapoint controls,
    TTS output, and an unload ("X") button to free up resources and reset the dropdown.
    """
    # Setup model and datapoint dropdown choices.
    model_configs = get_inference_configs("inference_configs/fixed_inference_image_resolution_589824")
    placeholder_model = "Select a YAML config file"
    # Set the autoload config
    autoload_config = "qwen2vl_lora_sft_lingoQA_scenery_25000.yaml"
    if autoload_config in model_configs:
        default_model = autoload_config
    else:
        default_model = placeholder_model
    model_dropdown_choices = [placeholder_model] + model_configs

    datapoints, datapoint_choices = get_datapoints("data/lingoQA_evaluation.json")
    placeholder_dp = "Select a datapoint"
    datapoint_dropdown_choices = [placeholder_dp] + datapoint_choices

    custom_css = """
    #image-container .column {
        max-height: 240px !important;
        overflow-y: auto;
    }
    .gallery-container {
        max-height: 240px !important;
        overflow-y: auto;
    }
    .audio-container {
        min-height: 212px;
    }
    """

    # JavaScript to trigger cleanup when the page unloads.
    cleanup_js = """
    <script>
      window.addEventListener("beforeunload", function(event) {
        navigator.sendBeacon("/cleanup", "");
      });
    </script>
    """

    with gr.Blocks(css=custom_css) as demo:
        # Inject cleanup JS.
        gr.HTML(cleanup_js)

        # ---------------------------
        # State Components
        # ---------------------------
        chatbot = gr.Chatbot(show_copy_button=True, type='messages')
        messages = gr.State([])
        uploaded_files = gr.State([])
        chat_instance = gr.State(None)  # holds the loaded StreamedChat instance

        # ---------------------------
        # Layout
        # ---------------------------
        with gr.Row():
            with gr.Column(scale=1):
                chatbot
                with gr.Row():
                    query = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message here...",
                        lines=10, max_lines=10
                    )
                    record_audio = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record"
                    )
                record_audio.change(fn=transcribe_audio, inputs=record_audio, outputs=[query, record_audio])
                read_out_toggle = gr.Checkbox(
                    value=False,
                    label="Read out response",
                    interactive=True
                )
            with gr.Column(scale=1):
                with gr.Row(elem_id="image-container"):
                    with gr.Column(scale=1):
                        image_upload = gr.File(label="Upload Images/Videos", file_count="multiple")
                    with gr.Column(scale=1):
                        image_gallery = gr.Gallery(label="Preview Uploads", columns=3)
                remove_images_btn = gr.Button("Remove All Uploads", variant="secondary")
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary", interactive=False)
        with gr.Row():
            clear_btn = gr.Button("Clear Conversation")
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=model_dropdown_choices,
                    label="Load Model",
                    info="Select a YAML config file from inference_configs",
                    interactive=True,
                    value=default_model   # autoload config if available
                )
            with gr.Column():
                datapoint_dropdown = gr.Dropdown(
                    choices=datapoint_dropdown_choices,
                    label="Load Datapoint",
                    info="Select a datapoint from JSON",
                    interactive=True,
                    value=placeholder_dp
                )
        # Row containing model status and the unload ("X") button.
        with gr.Row():
            model_status = gr.Markdown("No model loaded")
            unload_model_button = gr.Button("Clear VRAM", variant="secondary", interactive=False)
        
        audio_output = gr.Audio(label="Assistant Audio", interactive=False, autoplay=False, visible=True)

        # ---------------------------
        # Event Handlers / Bindings
        # ---------------------------
        def update_gallery(new_files, current_files):
            if new_files is None:
                new_files = []
            new_entries = []
            for file in new_files:
                filename = os.path.basename(file)
                new_entries.append((file, filename))
            all_entries = current_files + new_entries
            return all_entries, all_entries, None

        image_upload.change(
            fn=update_gallery,
            inputs=[image_upload, uploaded_files],
            outputs=[uploaded_files, image_gallery, image_upload]
        )
        
        def load_datapoint(selected_index, current_files):
            if selected_index == placeholder_dp:
                return current_files, gr.update(), gr.update()
            try:
                index = int(selected_index.split(" ")[1])
            except Exception:
                return current_files, gr.update(), gr.update()
            if index < 0 or index >= len(datapoints):
                return current_files, gr.update(), gr.update()
            dp = datapoints[index]
            new_entries = []
            for img in dp.get("images", []):
                filename = os.path.basename(img)
                new_entries.append((img, filename))
            question_text = dp.get("question", "").replace("<image>", "")
            return new_entries, new_entries, question_text

        datapoint_dropdown.change(
            fn=load_datapoint,
            inputs=[datapoint_dropdown, uploaded_files],
            outputs=[uploaded_files, image_gallery, query]
        )
        
        # --- Chained model-loading callbacks ---
        def disable_submit(selected_model):
            if selected_model == placeholder_model:
                return selected_model, "No model loaded", gr.update(interactive=False)
            return selected_model, f"Loading model: {selected_model}...", gr.update(interactive=False)

        def clear_garbage(selected_model):
            torch_gc()
            return selected_model

        def load_model(selected_model):
            if selected_model == placeholder_model:
                return None, "No model loaded", gr.update(interactive=False)
            model_path = os.path.join("inference_configs/fixed_inference_image_resolution_589824", selected_model)
            chat_inst = StreamedChat(model_path)
            return chat_inst, f"Model loaded: {selected_model}", gr.update(interactive=True)

        # When unloading the model, free VRAM, clear the chat instance,
        # hide the unload button, and reset the dropdown to the placeholder.
        def unload_model(chat_inst):
            if chat_inst is not None:
                torch_gc()
            return None, "No model loaded", gr.update(interactive=False), gr.update(interactive=False), gr.update(value=placeholder_model)

        # Helper to update the unload button's visibility.
        def show_unload_button(chat_inst):
            if chat_inst is not None:
                return gr.update(interactive=True)
            return gr.update(interactive=False)

        dummy_out = gr.Textbox(visible=False)
        # Chain triggered by model_dropdown change.
        model_dropdown.change(
            fn=disable_submit,
            inputs=[model_dropdown],
            outputs=[dummy_out, model_status, submit_btn]
        ).then(
            fn=clear_garbage,
            inputs=[dummy_out],
            outputs=[dummy_out]
        ).then(
            fn=load_model,
            inputs=[dummy_out],
            outputs=[chat_instance, model_status, submit_btn]
        ).then(
            fn=show_unload_button,
            inputs=[chat_instance],
            outputs=[unload_model_button]
        )
        
        # Bind the unload button to clear the model and reset the dropdown.
        unload_model_button.click(
            fn=unload_model,
            inputs=[chat_instance],
            outputs=[chat_instance, model_status, submit_btn, unload_model_button, model_dropdown]
        )
        
        # Also update the unload button visibility on page load (for autoloaded models)
        demo.load(
            fn=load_model,
            inputs=model_dropdown,
            outputs=[chat_instance, model_status, submit_btn]
        ).then(
            fn=show_unload_button,
            inputs=[chat_instance],
            outputs=[unload_model_button]
        )
        
        # --- Chat processing callback with streaming and TTS ---
        def process_chat(chatbot, messages, query, uploaded_files, chat_inst, read_out_toggle):
            if chat_inst is None:
                print("No chat model loaded!")
                return chatbot, messages, query, None

            print("User query:", query)
            image_paths = [file for file, _ in uploaded_files] if uploaded_files else []
            print("Images provided:", image_paths)
            
            # Add user message
            user_msg = {"role": "user", "content": query}
            messages = messages + [user_msg]
            chatbot = chatbot + [user_msg]
            yield chatbot, messages, query, None
            
            # Add empty assistant message
            assistant_msg = {"role": "assistant", "content": ""}
            messages = messages + [assistant_msg]
            chatbot = chatbot + [assistant_msg]
            yield chatbot, messages, query, None
            
            # Stream the response
            full_response = ""
            for partial in chat_inst.stream_chat(query, files=image_paths):
                full_response = partial
                messages[-1]["content"] = full_response
                chatbot[-1]["content"] = full_response
                # Only yield the updated chatbot without audio during streaming
                yield chatbot, messages, query, None
            
            # Always generate TTS after streaming is complete
            print("Generating audio for the last response...")
            tts_output_file = "audio_outputs/last_response.wav"
            tts_instance.say(
                full_response,
                output_file=tts_output_file,
                voice='af_heart',
                speed=1,
                split_pattern=r'\n+'
            )
            
            print("Outputting audio!")
            # Set autoplay based on the read_out_toggle
            audio_component = gr.update(value=tts_output_file, autoplay=read_out_toggle)
            # Final yield with audio
            yield chatbot, messages, query, audio_component
        
        submit_btn.click(
            fn=process_chat,
            inputs=[chatbot, messages, query, uploaded_files, chat_instance, read_out_toggle],
            outputs=[chatbot, messages, query, audio_output]
        )
        
        clear_btn.click(
            fn=lambda: ([], []),
            outputs=[chatbot, messages]
        )
        
        remove_images_btn.click(
            fn=clear_images,
            outputs=[uploaded_files, image_gallery]
        )
        
        # ---------------------------
        # Add a cleanup endpoint for VRAM cleanup on unload.
        # ---------------------------
        def cleanup_endpoint():
            torch_gc()
            print("Cleanup triggered: VRAM freed.")
            return {"status": "cleanup done"}
        
        demo.app.add_api_route("/cleanup", cleanup_endpoint, methods=["POST"])
        
    components = {
        "chatbot": chatbot,
        "messages": messages,
        "uploaded_files": uploaded_files,
        "query": query,
        "submit_btn": submit_btn,
        "image_upload": image_upload,
        "image_gallery": image_gallery,
        "model_dropdown": model_dropdown,
        "datapoint_dropdown": datapoint_dropdown,
        "model_status": model_status,
        "clear_btn": clear_btn,
        "chat_instance": chat_instance,
        "read_out_toggle": read_out_toggle,
        "audio_output": audio_output,
        "unload_model_button": unload_model_button,
    }
    return demo, components


# ======================================
# Example usage.
# ======================================
if __name__ == "__main__":
    demo, components = create_chat_ui()
    demo.launch(allowed_paths=["data/binaries/lingoQA_evaluation/images/val"], share=True)
    # demo.launch(server_port=7861, allowed_paths=["data/binaries/lingoQA_evaluation/images/val"], share=False)
    # demo.launch(allowed_paths=["data/binaries/lingoQA_evaluation/images/val"], share=False)