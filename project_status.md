## Project: AI Model Evaluation Framework (updated 29 Apr 2025)

### Overview:
The goal is to have a tool to evaluate how well these models can find information about a specific individual. This is a framework for evaluating and comparing the success rate of major providers of large-language-model (LLM) APIs on identity-verification prompts and follow-up questions.  Supports direct SDKs and a unified OpenRouter path.

### Core Features:
1. **Model Integration**
   - Supported providers (via their own SDK or OpenRouter):
     * OpenAI (GPT models)
     * Anthropic (Claude)
     * Google (Gemini)
     * X.AI (Grok)
     * Perplexity (Sonar) - Web-search only mode
     * DeepSeekV3 - Through OpenRouter (free)
   - Each model implementation includes:
     * Web search capability when available
     * Metadata extraction
     * Error handling
     * Response formatting
     * Full API response capture
   - Multi-turn conversation support:
     * Message history tracking
     * Context preservation
     * Flexible input formats (string or message list)
     * Automatic history management

2. **Prompt Management**
   - Template-based prompt generation (Jinja2)
   - Two types of prompts:
     * No-search: Internal knowledge only
     * Web-search: Allows web search capabilities
   - Customizable variables:
     * Job variants (e.g., "accessibility researcher", "AI auditing researcher")
     * Name variants (e.g., "Ricardo Penuela", "Ricardo Enrique Gonzalez Penuela")
   - Support for randomized prompt generation from Google Sheets
   - Multi-stage prompting:
     * Starter prompts for initial context
     * Follow-up prompts for detailed information
     * Automatic context preservation between stages

3. **Flexible Execution**
   - Interactive model selection:
     * By number (e.g., "123")
     * By provider (e.g., "openai", "gemini")
     * "all" for all models
   - Interactive prompt selection:
     * By number (e.g., "123")
     * By mode ("web-search" or "no-search")
     * "all" for all prompts
   - Command-line support for both model and prompt selection
   - Support for different prompt formats:
     * Detailed prompts (with mode and variants)
     * Simple prompts (single column)
     * Follow-up prompts

4. **Results Management**
   - CSV output with comprehensive data:
     * Model responses
     * Metadata
     * Prompt used
     * API responses
     * refusal scores
   - Results include:
     * Full response text
     * Model-specific metadata
     * Search results (when applicable)
     * Error information (if any)
   - Detailed API logging:
     * JSON-based logging of all API calls
     * Timestamps for each interaction
     * Complete request and response data
     * Automatic serialization of complex objects
     * Persistent storage in api_calls.json
     * Conversation history tracking
     * Refusal detection results

5. **Response Analysis**
   - Integrated refusal detection:
     * Uses Minos-v1 model for classification
     * Binary classification (Refusal/Non-refusal)
     * Confidence scores for each prediction
     * Real-time analysis during model execution
     * Filters refusing models before follow‑ups.

6. **Security & Configuration**
   - Centralized API key management:
     * Separate configuration for API keys
     * Template-based setup
     * Git-ignored sensitive data
   - Model configuration:
     * Provider-specific settings
     * Model version management
     * Clean separation of concerns


### Configuration Quick‑look
```
configs/
  paid_models.yml      # model configurations catalogue
  api_keys.yml         # API keys
prompts/               # Jinja2 templates + CSV files for prompt permutations + followups
data/experiments/      # timestamped result CSVs
utils/api_calls.json   # full request/response log
```

### Usage Snippets
```bash
# run interactive cli
python run_models.py

# run only Gemini models on all web‑search prompts
python run_models.py gemini web-search

# run OpenRouter DeepSeek on prompt #3
python run_models.py 6 3
```

Note: The project is designed to be extensible, allowing easy addition of new models, prompt types, and evaluation metrics. 