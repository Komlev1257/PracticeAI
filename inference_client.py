from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceHTTPClient(
    api_url=os.getenv("API_URL"),
    api_key=os.getenv("API_KEY")
)
MODEL_ID = os.getenv("MODEL_ID")
