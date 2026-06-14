import os

from dotenv import load_dotenv

# Load variables from a .env file (if present) into the process environment.
load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Default model for all nodes; override via ANTHROPIC_MODEL in .env.
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
