import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)


class LLMClient:

    def __init__(self):
        self.model = genai.GenerativeModel("models/gemini-3.1-flash-lite-preview")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text