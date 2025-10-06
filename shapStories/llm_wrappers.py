from abc import ABC, abstractmethod
# from openai import OpenAI
from google import genai
from google.genai import types
import replicate
import requests

# from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWrapper(ABC):
    @abstractmethod
    def generate_response(self, prompt):
        """
        Generates a response to the given prompt.

        :param prompt: The input prompt to generate a response for.
        :return: The generated response as a string.
        """
        pass


class GptApi(LLMWrapper):

    def __init__(self, api_key, model):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, prompt):

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a teacher, skilled at explaining complex AI decisions to general audiences."},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content


class GeminiAPI(LLMWrapper):

    def __init__(self, api_key, model):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate_response(self, prompt):
        return self.client.models.generate_content(model=self.model, contents=prompt).text


class OllamaWrapper(LLMWrapper):

    def __init__(self, model):
        self.model = model

    def generate_response(self, prompt):
        api_endpoint = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {"prompt": prompt, "temperature": 0.1, "stream": False, "model": self.model}
        response = requests.post(api_endpoint, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to generate response: {response.text}")
        response_json = response.json()
        if response_json.get("done_reason") != "stop":
            raise Exception(f"Generation did not stop: {response_json}")

        return response_json["response"]
        
# Not advised to use


class HfGemma(LLMWrapper):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

    def generate_response(self, prompt, max_tokens=1500):
        input_ids = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(**input_ids, max_new_tokens=max_tokens)
        print(self.tokenizer.decode(outputs[0]))


class LlamaAPI():
    def __init__(self, api_key, model="meta/meta-llama-3-70b-instruct"):
        self.api_key = api_key
        self.model = model
        self.client = replicate.Client(api_key)

    def generate_response(self, prompt, max_tokens=512):
        output = self.client.run(
            self.model,
            input={
                "top_p": 0.9,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "min_tokens": 0,
                "temperature": 0.6,
                "prompt_template": "system\n\nYou are a helpful assistantuser\n\n{prompt}assistant\n\n",
                "presence_penalty": 1.15,
                "frequency_penalty": 0.2
            }
        )
        response_text = ""
        for item in output:
            response_text += str(item)
        return response_text
