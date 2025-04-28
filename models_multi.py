import openai
from openai import OpenAI
import anthropic
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import requests
import yaml
from pathlib import Path
from typing import Union, List, Dict, Tuple


def load_api_keys():
    """Load API keys from the api_keys.yml file."""
    api_keys_path = Path(__file__).parent / "configs" / "api_keys.yml"
    if not api_keys_path.exists():
        raise FileNotFoundError(
            "api_keys.yml not found. Please add api_keys.yml to configs "
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

    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        mode: str
    ) -> Tuple[str, Dict, Dict, List[Dict[str, str]]]:
        """
        Accept either a prompt string or a message-history list.
        Normalize to a messages list, call _call_model, then append the assistant response.
        Returns: text, metadata, raw_response, updated_messages
        """
        # Normalize input
        if isinstance(prompt_or_messages, str):
            messages: List[Dict[str, str]] = [{"role": "user", "content": prompt_or_messages}]
        else:
            # copy so we don’t mutate the caller’s list
            messages = list(prompt_or_messages)

        # Call the provider-specific implementation
        text, metadata, raw = self._call_model(messages, mode)

        # Append assistant turn to history
        messages.append({"role": "assistant", "content": text})

        return text, metadata, raw, messages

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str
    ) -> Tuple[str, Dict, Dict]:
        raise NotImplementedError

# OpenAI implementation
class OpenAIModel(BaseModel):
    def __init__(self, name: str, model: str):
        super().__init__(name)
        openai.api_key = API_KEYS["openai"]["api_key"]
        self.model = model

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str
    ) -> Tuple[str, Dict, Dict]:
        client = OpenAI()
        params: Dict[str, object] = {
            "model": self.model,
            # `input` accepts a list of message dicts for multi-turn
            "input": messages,
            "temperature": 0.2,
            "user": "1"
        }
        # include web-search tool if requested
        if mode == "web-search":
            params["tools"] = [{"type": "web_search_preview"}]


        ## Alter prompt with extra content for starter prompt for testing
        #params["input"][0]['content'] = params["input"][0]['content'] + "from Cornell University" 
        ## Check information being sent.
        print(params)
        response = client.responses.create(**params)
        # extract output text and metadata
        text = response.output_text
        metadata = getattr(response, "output", {})
        return text, metadata, response

# Other model classes remain unchanged below
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
        metadata = getattr(resp, 'metadata', {})
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
        text = "".join(part.text for part in response.candidates[0].content.parts)
        metadata = {}
        if hasattr(response.candidates[0], 'grounding_metadata') and hasattr(response.candidates[0].grounding_metadata, 'search_entry_point'):
            metadata['search_content'] = response.candidates
        return text, metadata , response

class GrokModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        self.client = OpenAI(
            api_key=API_KEYS["grok"]["api_key"],
            base_url="https://api.x.ai/v1"
        )
        self.model  = model

    def generate(self, prompt: str, mode: str) -> tuple[str, dict, dict]:
        messages = [
            {"role": "user",   "content": prompt}
        ]
        kwargs = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  1024,
            "temperature": 0.0
        }
        completion = self.client.chat.completions.create(**kwargs)
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
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(self.url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            metadata = {'model': data.get('model'), 'usage': data.get('usage'), 'id': data.get('id')}
            return text, metadata, data
        except Exception as e:
            return f"[ERROR] {e}", {}, {}

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
