# Configuration file for Life Satisfaction AI Assistant
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

# Fallback Settings (when API is not available)
USE_FALLBACK_RESPONSES = True  # Set to True to use local responses when API fails

# Application Settings
APP_TITLE = "Life Satisfaction AI Assistant"
APP_ICON = "🌱"
PAGE_LAYOUT = "wide"

# Model Settings
MODEL_FILES = {
    "model": "life_satisfaction_model.pkl",
    "scaler": "scaler.pkl", 
    "label_encoder": "label_encoder.pkl",
    "X_columns": "X_columns.pkl"
}

# Chatbot Settings
CHATBOT_TEMPERATURE = 0.7
CHATBOT_MAX_TOKENS = 1000
CHATBOT_MODEL = "deepseek-chat"

# File Storage
PROGRESS_DIR = "user_progress"
PROGRESS_FILE_PREFIX = "progress_"

# UI Settings
MAX_PROGRESS_FILES_DISPLAY = 5
