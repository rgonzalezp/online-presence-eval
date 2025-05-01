import openai
from openai import OpenAI
import anthropic
from google import genai
from google.genai.types import Content, Part, Tool, GenerateContentConfig, GoogleSearch
from google.genai import errors
import requests
import yaml
from pathlib import Path
from typing import Union, List, Dict, Tuple
import time, threading
from collections import deque


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

    _LOCK          = threading.Lock()
    _WINDOWS       = {}   
    _RATE_LIMITS   = {}


    @classmethod
    def set_rate_limit(cls, model_name: str, rpm: int):
        """Register or update a requests-per-minute limit for a model slug."""
        with cls._LOCK:
            cls._RATE_LIMITS[model_name] = rpm
            cls._WINDOWS.setdefault(model_name, deque())

    def __init__(self, name: str):
        self.name = name
        BaseModel.set_rate_limit(self.name, BaseModel._RATE_LIMITS.get(self.name, 10))

    def _block_if_needed(self):
        """
        Ensure this model stays below its RPM.
        Called immediately before every network request.
        """
        rpm   = BaseModel._RATE_LIMITS[self.name]
        win   = BaseModel._WINDOWS[self.name]
        now   = time.time()

        # drop timestamps older than 60 s
        while win and now - win[0] >= 60:
            win.popleft()

        if len(win) >= rpm:
            sleep_for = 60 - (now - win[0]) + 0.01  # small buffer
            print(f"[{self.name}] ⏳  rate limit reached. Sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
            # after sleep, purge again
            now = time.time()
            while win and now - win[0] >= 60:
                win.popleft()

        win.append(now)

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

        self._block_if_needed()

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
            "temperature": 0.1115,
            "user": "1",
            "max_output_tokens": 1024
        }
        # include web-search tool if requested
        if mode == "web-search":
            params["tools"] = [{"type": "web_search_preview"}]


        ## Alter prompt with extra content for starter prompt for testing
        #params["input"][0]['content'] = params["input"][0]['content'] + "from Cornell University" 

        try:
            resp = client.responses.create(**params)
        except Exception as e:
            return f"[ERROR] OpenAI request failed ({e})", {}, {}
        text = getattr(resp, "output_text", None)
        if not text:
            return "[ERROR] Empty response from OpenAI", {}, resp.__dict__

        metadata = getattr(resp, "output", {})
        return text.strip(), metadata, resp

class AnthropicModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        
        self.client = anthropic.Anthropic( api_key = API_KEYS["anthropic"]["api_key"])
        self.model  = model

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str
    ) -> Tuple[str, Dict, Dict]:
        resp = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1115
        )
        metadata = {
            "model":     getattr(resp, "model", None),
            "usage":     getattr(resp, "usage", {}),
            "id":        getattr(resp, "id", None),
        }

        text = "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        ).strip()

        return text, metadata, resp

class GeminiModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        self.client = genai.Client(api_key=API_KEYS["gemini"]["api_key"])
        self.model  = model

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str
    ) -> Tuple[str, Dict, Dict]:
        tools = []

        content_objects = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                print(f"Warning: Skipping invalid message format at index {i}: {msg}")
                continue

            role = 'model' if msg.get('role', 'user').lower() in ['assistant', 'model'] else 'user'
            content_text = msg.get('content')

            # Ensure content is string
            if content_text is None:
                 part_text = ""
            elif not isinstance(content_text, str):
                 part_text = str(content_text)
            else:
                 part_text = content_text

            try:
                 part_object = Part(text=part_text)
                 content_object = Content(role=role, parts=[part_object])
                 content_objects.append(content_object)
            except NameError:
                 # Fallback or error if Content/Part are NOT actually in google.genai.types
                 print("ERROR: Cannot find 'Content' or 'Part' in 'google.genai.types'. "
                       "Ensure your import path and SDK are correct.")
                 # You might return an error or raise an exception here
                 return "Error: SDK type mismatch", {}, {}
            except Exception as e:
                 print(f"Error creating Content/Part object: {e}")
                 return f"Error: SDK object creation failed ({e})", {}, {}
            
        ## When Gemini does not have access to web search, it sticks to the conversation so far and it does not provide any information that is not available there. So for now we flag that it can always use web-search if it's asking follow-up questions. Will probably have to change this later to handle the availability of the tool or not in an easier way
        if mode == "web-search" or len(messages) > 1:
            tools = [Tool(google_search=GoogleSearch())]
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=content_objects,
                config=GenerateContentConfig(
                    tools=tools,
                    response_modalities=["TEXT"],
                    max_output_tokens=1024,
                    temperature=0.1115
                )
            )
        except errors.APIError as e:
            print(e.code) # 404
            print(e.message)
            
        # ---------- sanity-check the response structure ---------------
        if not getattr(response, "candidates", None):
            return "[ERROR] Gemini returned no candidates", {}, response

        cand = response.candidates[0]
        if not getattr(cand, "content", None):
            return "[ERROR] Candidate had no content", {}, response

        parts = getattr(cand.content, "parts", None)
        if not parts:
            return "[ERROR] Candidate content.parts empty", {}, response

        # gather text safely
        text_chunks = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
        if not text_chunks:
            return "[ERROR] No text parts in response", {}, response

        text = "".join(text_chunks).strip()
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

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str
    ) -> Tuple[str, Dict, Dict]:

        kwargs = {
            "model":       self.model,
            "messages":    messages,        # full chat history
            "temperature": 0.1115,
            "max_tokens":  1024,
        }
        # (Grok doesn’t expose web-search yet; add tool args here when it does)

        resp = self.client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content.strip()
        metadata = {
            "model":              getattr(resp, "model", None),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "usage":              getattr(resp, "usage", {}),
        }
        return text, metadata, resp

class PerplexityModel(BaseModel):
    def __init__(self, name, model):
        super().__init__(name)
        self.api_key = API_KEYS["perplexity"]["api_key"]
        self.model = model
        self.url = "https://api.perplexity.ai/chat/completions"

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str
    ) -> Tuple[str, Dict, Dict]:
        if mode != "web-search" and len(messages) == 1:
            return "Perplexity model only works in web-search mode", {}, {}
        payload = {
            "model": self.model,
            "messages": messages
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

class OpenRouterModel(BaseModel):
    """Universal wrapper for any OpenRouter model slug."""
    def __init__(self, name: str, model: str):
        super().__init__(name)
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEYS["openrouter"]["api_key"]
        )

    def _call_model(
        self,
        messages: List[Dict[str, str]],
        mode: str
    ) -> Tuple[str, Dict, Dict]:
        
        needs_online = mode == "web-search" or len(messages) > 1
        model_name   = f"{self.model}:online" if needs_online else self.model

        resp = self.client.chat.completions.create(
            model= model_name,          
            messages=messages,
            temperature=0.1115,
            max_tokens=1024,
        )
        print(resp)
        text = resp.choices[0].message.content
        metadata = dict(
            usage=getattr(resp, "usage", {}),
            id=resp.id,
            provider="openrouter"
        )
        
        return text, metadata, resp


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
        elif prov == "openrouter":                        
            models[name] = OpenRouterModel(name, cfg["model"])
        else:
            raise ValueError(f"Unknown provider: {prov}")
    return models
