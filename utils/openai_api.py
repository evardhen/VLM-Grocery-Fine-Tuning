import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
import yaml

def send_openai_request(image_path, prompt_path, category = None, model="gpt-4o"):
    try:
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        client = OpenAI(api_key=api_key)
        base64_image = encode_image(image_path)

        with open(prompt_path, "r") as file:
            prompt = yaml.safe_load(file)

        prompt = prompt["simple_prompt"]
        if category != None:
            prompt = prompt.format(category=category)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
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