import whisper
import speech_recognition as sr
import tempfile

class WhisperTranscriber:
    """
    A simple class that loads a specified Whisper model and transcribes a given audio file.
    """
    def __init__(self, model_name: str = "base"):
        """
        :param model_name: Name of the Whisper model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
        :param audio_path: Path to the audio file you want to transcribe.
        """
        self.model_name = model_name
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))



    def run(self, audio_path):
        """
        Loads the Whisper model, transcribes the audio at self.audio_path, and prints the resulting text.
        """
        # Load the specified Whisper model
        model = whisper.load_model(self.model_name)
        
        # Transcribe the audio file
        result = model.transcribe(audio_path)
        
        # Print the text result
        print("Transcribed Text:\n", result["text"])
        return result["text"]

    def transcribe_with_speech_recognition(self):
        """
        Uses the SpeechRecognition library to record from the default microphone
        until it detects speech has ended. Prints the recognized text.

        Note: This uses the Google Speech API by default, which requires an internet connection.
              It also detects end-of-speech automatically.
        """
        # Create a recognizer
        recognizer = sr.Recognizer()

        # Use the default system microphone as the audio source
        with sr.Microphone() as source:
            # Optionally, adjust for ambient noise if needed:
            # recognizer.adjust_for_ambient_noise(source, duration=1)

            print("SpeechRecognition: Please speak now.")
            audio_data = recognizer.listen(source)  # automatically stops when speech ends
            print("Recording stopped. Processing...")

        # Convert SpeechRecognition's AudioData to WAV bytes
        wav_bytes = audio_data.get_wav_data()

        # Write these bytes to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(wav_bytes)
            tmp_file.flush()
            temp_wav_path = tmp_file.name

        # Load the specified Whisper model
        model = whisper.load_model(self.model_name)

        # Transcribe the temporary file
        result = model.transcribe(temp_wav_path)
        print("Whisper Transcription (Microphone):\n", result["text"])

if __name__ == "__main__":
    # Example usage:
    transcriber = WhisperTranscriber(model_name="base")

    # 2. Transcribe from mic using Whisper (offline)
    transcriber.transcribe_with_speech_recognition()
