import numpy as np
from kokoro import KPipeline
import soundfile as sf
import os

class KokoroTTS:
    """
    A wrapper class for the Kokoro text-to-speech model.
    
    Example usage:
    
        from kokoro_tts import KokoroTTS
        
        tts = KokoroTTS(lang_code='a')
        tts.say(
            text="Your input text goes here.",
            output_file="output.wav",
            voice="af_heart",
            speed=1,
            split_pattern=r'\n+'
        )
    """
    
    def __init__(self, lang_code: str = 'a', sample_rate: int = 24000):
        """
        Initialize the Kokoro TTS model.
        
        Args:
            lang_code (str): Language code to use. For example:
                'a' for American English,
                'b' for British English,
                'j' for Japanese (requires misaki[ja]),
                'z' for Mandarin Chinese (requires misaki[zh]).
            sample_rate (int): Sample rate for the generated audio. Defaults to 24000.
        """
        print("Initializing Kokoro pipeline...")
        self.sample_rate = sample_rate
        self.pipeline = KPipeline(lang_code=lang_code)
        print("Kokoro pipeline loaded.")

    def say(self,
            text: str,
            output_file: str = "output.wav",
            voice: str = 'af_heart',
            speed: float = 1,
            split_pattern: str = r'\n+'):
        """
        Synthesize speech from the provided text and save the audio as a WAV file.
        
        Args:
            text (str): The input text to convert to speech.
            output_file (str): The file path where the output WAV file will be saved.
            voice (str): The voice identifier to use. Change as needed.
            speed (float): The speech speed factor.
            split_pattern (str): Regular expression pattern used to split the input text into segments.
        """
        print("Generating audio...")
        # Generate audio segments. Each item in the generator is a tuple:
        # (graphemes, phonemes, audio segment as numpy array)
        generator = self.pipeline(
            text, 
            voice=voice, 
            speed=speed, 
            split_pattern=split_pattern
        )
        
        audio_segments = []
        for idx, (gs, ps, audio) in enumerate(generator):
            print(f"Segment {idx}:")
            print("  Graphemes:", gs)
            print("  Phonemes: ", ps)
            audio_segments.append(audio)
        
        if not audio_segments:
            print("No audio segments generated.")
            return
        
        # Concatenate all audio segments into one array
        full_audio = np.concatenate(audio_segments)

        # Create the directory if it does not exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the combined audio to the output file
        sf.write(output_file, full_audio, self.sample_rate)
        print(f"Audio saved to {output_file}")

