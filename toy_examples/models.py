import openai
from openai import OpenAI
import anthropic
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import requests
import yaml
from pathlib import Path
from typing import Union, List, Tuple, Dict


def load_api_keys():
    """Load API keys from the api_keys.yml file."""
    api_keys_path = Path(__file__).parent / "configs" / "api_keys.yml"
    if not api_keys_path.exists():
        raise FileNotFoundError(
            "api_keys.yml not found. Please add api_keys.yml to \configs "
            "and fill in your API keys."
        )
    
    with open(api_keys_path) as f:
        return yaml.safe_load(f)

# Load API keys once at module level
API_KEYS = load_api_keys()

# Base interface
class BaseModel:
    def __init__(self, name: str):
        self.name = name
    def generate(self, prompt: str) -> tuple[str, dict]:
        """Generate a response to a prompt.
        
        Returns:
            tuple[str, dict]: The generated text and metadata/annotations
        """
        raise NotImplementedError

class OpenAIModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        openai.api_key = API_KEYS["openai"]["api_key"]
        self.model = model

    def generate(self, prompt: str, mode: str) -> tuple[str, dict, dict]:
        # instantiate the XAI-compatible OpenAI client
        client = OpenAI()

        # base args for every call
        params = {
            "model": self.model,
            "input": prompt,
            "temperature": 0.3,
            "user": "1"
        }

        # only include the web-search tool if mode == "web-search"
        if mode == "web-search":
            params["tools"] = [{"type": "web_search_preview"}]

        response = client.responses.create(**params)
        
        # Extract annotations if available
        metadata = response.output
        
        if hasattr(response, 'output') and hasattr(response.output, 'content'):
            annotations = response[1].content[0].annotations
        
        return response.output_text, metadata, response

class AnthropicModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        anthropic.api_key = API_KEYS["anthropic"]["api_key"]
        self.client = anthropic.Client()
        self.model  = model

    def generate(self, prompt, mode=None) -> tuple[str, dict, dict]:
        resp = self.client.completions.create(
            model=self.model,
            prompt=anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT,
            max_tokens_to_sample=1024,
            temperature=0.0
        )
        
        # Extract any available metadata
        metadata = {}
        if hasattr(resp, 'metadata'):
            metadata = resp.metadata
            
        return resp.completion.strip(), metadata, resp

class GeminiModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        
        self.client = genai.Client(api_key=API_KEYS["gemini"]["api_key"])
        self.model  = model

    def generate(self, prompt: str, mode: str) -> tuple[str, dict, dict]:
        tools = []
        if mode == "web-search":
            tools = [Tool(google_search=GoogleSearch())]

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=GenerateContentConfig(
                tools=tools,
                response_modalities=["TEXT"],
            )
        )
        
        # Extract text from response
        text = "".join(part.text for part in response.candidates[0].content.parts)
        
        # Extract grounding metadata if available
        metadata = {}
        if hasattr(response.candidates[0], 'grounding_metadata') and \
           hasattr(response.candidates[0].grounding_metadata, 'search_entry_point'):
            metadata['search_content'] = response.candidates
        
        return text, metadata , response

class GrokModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        # use the OpenAI-compatible XAI client
        self.client = OpenAI(
            api_key=API_KEYS["grok"]["api_key"],
            base_url="https://api.x.ai/v1"
        )
        self.model  = model

    def generate(self, prompt: str, mode: str) -> tuple[str, dict, dict]:
        messages = [
            {"role": "system", "content": "You are a PhD-level researcher."},
            {"role": "user",   "content": prompt}
        ]
        kwargs = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  1024,
            "temperature": 0.0
        }
        if mode == "web-search":
            # x.ai/grok currently does not support external tools,
            # but if it did, you could add e.g.:
            # kwargs["tools"] = [{"type": "web_search_preview"}]
            pass

        completion = self.client.chat.completions.create(**kwargs)
        
        # Extract metadata if available
        metadata = {}
        if hasattr(completion, 'model') or hasattr(completion, 'system_fingerprint'):
            metadata = {
                'model': getattr(completion, 'model', None),
                'system_fingerprint': getattr(completion, 'system_fingerprint', None)
            }
            
        return completion.choices[0].message.content.strip(), metadata, completion

class PerplexityModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        self.api_key = API_KEYS["perplexity"]["api_key"]
        self.model = model
        self.url = "https://api.perplexity.ai/chat/completions"

    def generate(self, prompt: str, mode: str) -> tuple[str, dict, dict]:
        if mode != "web-search":
            return "Perplexity model only works in web-search mode", {}, {}
        

        ## Do not add any system prompt, we want to have similar conditions for all models that mimic user usage.
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract text from response
            text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Extract metadata
            metadata = {
                'model': response_data.get('model'),
                'usage': response_data.get('usage'),
                'id': response_data.get('id')
            }
            
            return text, metadata, response_data
            
        except Exception as e:
            return f"[ERROR] {str(e)}", {}, {}

def create_models(model_configs):
    models = {}
    for cfg in model_configs:
        name, prov = cfg["name"], cfg["provider"]
        if prov == "openai":
            models[name] = OpenAIModel(name, cfg["model"])
        elif prov == "anthropic":
            models[name] = AnthropicModel(name, cfg["model"])
        elif prov == "gemini":
            models[name] = GeminiModel(name, cfg["model"])
        elif prov == "grok":
            models[name] = GrokModel(name, cfg["model"])
        elif prov == "perplexity":
            models[name] = PerplexityModel(name, cfg["model"])
        else:
            raise ValueError(f"Unknown provider: {prov}")
    return models 