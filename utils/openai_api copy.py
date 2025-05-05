import os
from dotenv import load_dotenv
from openai import OpenAI
import base64

def send_openai_request(dataset_entry, model="gpt-4o"):
    try:
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        client = OpenAI(api_key=api_key)

        # Prepare content with question + images
        content = [{"type": "text", "text": dataset_entry["question"]}]
        for image_path in dataset_entry["images"]:
            base64_image = encode_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                }
            )

        messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        
        response = client.chat.completions.create(
            messages=messages,
            model= model,
            temperature=0,
        )
        return response.choices[0].message.content
    
    except Exception as e:
        raise RuntimeError(f"Error in send_openai_request for {image_path}: {e}")
    
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(f"File not found: {image_path}") from fnfe
    except Exception as e:
        raise RuntimeError(f"Error encoding image {image_path}: {e}")