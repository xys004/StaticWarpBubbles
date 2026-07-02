"""
llm_wrapper.py
==============
A clean wrapper to call Gemini API interchangeably between local Execution and Google Colab.
"""

import os
import traceback

def get_gemini_api_key():
    """Detects where it's running to pull the API key safely."""
    # 1. First try system ENV variable (Works for local terminals)
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
        
    # 2. If missing, check if running in Google Colab 
    # (Colab uses google.colab.userdata for safe secret storage)
    try:
        from google.colab import userdata
        key = userdata.get('GEMINI_API_KEY')
        if key:
            return key
    except ImportError:
        pass
        
    raise ValueError("ERROR: GEMINI_API_KEY no detectada. Por favor configurala en el entorno local o en los secretos de Google Colab.")

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Invokes the Gemini Model according to config."""
    from google import genai
    from google.genai import types
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import GEMINI_MODEL
    
    api_key = get_gemini_api_key()
    client = genai.Client(api_key=api_key)
    
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def _do_call():
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7
            ),
        )
        return response.text
        
    try:
        return _do_call()
    except Exception as e:
        print(f"[!] Error calling LLM persistently: {e}")
        return ""
