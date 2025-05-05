import sys
from llamafactory.chat.chat_model import ChatModel
import os
from utils.video_processor import VideoFrameExtractor


class StreamedChat:
    def __init__(self, model_config_path: str):
        """
        Initializes the chat model with the provided configuration.
        
        Parameters:
            model_config_path (str): Path to the YAML config file.
        """
        self.model_config_path = model_config_path
        # Pass the configuration path to the model
        sys.argv[1:] = [self.model_config_path]
        self.chat_model = ChatModel()

    def chat(self, query: str, files=None) -> str:
        """
        Returns a full chat response given a text query and optional image/video paths.
        
        Parameters:
            query (str): The text prompt for the chat model.
            files (list, optional): List of file paths (images, videos).
        
        Returns:
            str: The complete response text.
        """
        if files is None:
            files = []
            
        # Define supported file extensions
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
        
        images = []
        videos = []
        
        # Process each file based on its extension
        for file_path in files:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            if ext in image_extensions:
                # Directly add image file paths (assumed to be acceptable as str)
                images.append(file_path)
            # In your StreamedChat.chat method:
            elif ext in video_extensions:
                extractor = VideoFrameExtractor(file_path, extraction_fps=1)  # 1 frame per second
                video_frames = extractor.extract_frames()  # Already JPEG-encoded bytes
                images.extend(video_frames)        
        messages = [{"role": "user", "content": query}]
        result = self.chat_model.chat(
            messages=messages,
            images=images,
            videos=videos,
            top_k = 1,
            temperature = 0.001
        )
        return result[0].response_text

    def stream_chat(self, query: str, files=None):
        """
        Streams a chat response token-by-token.
        In this example, the full response is first obtained, then we yield an
        ever-growing partial result word-by-word.
        
        Parameters:
            query (str): The text prompt for the chat model.
            files (list, optional): List of file paths (images, videos).
        
        Yields:
            str: The current accumulated response.
        """
        full_response = self.chat(query, files=files)
        tokens = full_response.split()
        current_text = ""
        for token in tokens:
            current_text += token + " "
            # Yield the current response (strip trailing space)
            yield current_text.strip()

if __name__ == "__main__":
    # Replace with your model configuration path
    model_config = "inference_configs/qwen2vl_lora_sft_lingoQA_scenery_1000.yaml"
    
    # Create an instance of the chat interface
    chat_interface = StreamedChat(model_config)
    
    # Example chat query with both images and videos.
    response = chat_interface.chat(
        "test", 
        images=["data/binaries/lingoQA_evaluation/images/val/0ad3c6571f5aff651d43b4014b693e94/0.jpg",
                "data/binaries/lingoQA_evaluation/images/val/0ad3c6571f5aff651d43b4014b693e94/1.jpg"],
        videos=["data/binaries/mllm_videos_demo/1.mp4"]
    )
    print("Full Response:", response)
    
    # Demonstrate streaming.
    print("Streaming response:")
    for partial in chat_interface.stream_chat(
        "test", 
        images=["data/binaries/lingoQA_evaluation/images/val/0ad3c6571f5aff651d43b4014b693e94/0.jpg",
                "data/binaries/lingoQA_evaluation/images/val/0ad3c6571f5aff651d43b4014b693e94/1.jpg"],
        videos=["data/binaries/mllm_videos_demo/1.mp4"]
    ):
        print(partial)
