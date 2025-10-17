from abc import ABC, abstractmethod
from google import genai
import json
import requests

class LLMWrapper(ABC):
    def __init__(self, temperature=0.1) -> None:
        self.temperature = temperature

    @abstractmethod
    def generate_response(self, prompt):
        """
        Generates a response to the given prompt.

        :param prompt: The input prompt to generate a response for.
        :return: The generated response as a string.
        """
        pass

class GeminiAPI(LLMWrapper):

    def __init__(self, api_key, model, temperature=0.1):
        super().__init__(temperature)
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate_response(self, prompt):
        return self.client.models.generate_content(model=self.model, contents=prompt).text


class OllamaWrapper(LLMWrapper):

    def __init__(self, model, temperature=0.1, host="localhost", port=11434):
        super().__init__(temperature)
        self.model = model
        self.port = port
        self.host = host

    def generate_response(self, prompt):
        api_endpoint = f"http://{self.host}:{self.port}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {"prompt": prompt, "temperature": self.temperature, "stream": False, "model": self.model}
        response = requests.post(api_endpoint, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to generate response: {response.text}")
        response_json = response.json()
        if response_json.get("done_reason") != "stop":
            raise Exception(f"Generation did not stop: {response_json}")

        return response_json["response"]
    
    def generate_json_response(self, prompt):
        api_endpoint = f"http://{self.host}:{self.port}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt, 
            "temperature": self.temperature, 
            "stream": False, 
            "model": self.model,
            "format": "json"
        }
        response = requests.post(api_endpoint, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to generate response: {response.text}")
        response_json = response.json()
        if response_json.get("done_reason") != "stop":
            raise Exception(f"Generation did not stop: {response_json}")

        try:
            return json.loads(response_json["response"])
        except json.decoder.JSONDecodeError:
            raise Exception(f"Failed to parse response: {response_json['response']}")
        