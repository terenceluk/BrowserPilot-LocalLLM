"""
Configuration for LM Studio Browser Agent.
Adjust these settings to match your LM Studio setup.
"""

# LM Studio API settings
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"  # LM Studio doesn't require a real key

# Model name — must match what's loaded in LM Studio
# Check LM Studio's model identifier (shown in the server tab)
# Options on this machine:
#   "qwen/qwen3.6-35b-a3b"       <- RECOMMENDED: fast MoE, good tool calling
#   "gpt-oss-120b"           <- slower but capable
#   "qwen3-coder-next"       <- code-focused, less ideal for browsing
#MODEL_NAME = "qwen/qwen3.5-35b-a3b"
MODEL_NAME = "qwen3.6-35b-a3b"
#MODEL_NAME = "openai/gpt-oss-120b"
SHOW_LLM_CONSOLE_TRACE = True  # Print raw LM Studio content and reasoning_content in the terminal

# Browser agent settings
BROWSER_HEADLESS = False  # Set True to run browser in background
MAX_STEPS = 50  # Max actions the agent can take per task
USE_VISION = False  # Set True if your model supports vision (multimodal)

# Browser settings
BROWSER_WIDTH = 1280
BROWSER_HEIGHT = 1200
