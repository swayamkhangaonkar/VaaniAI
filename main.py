import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory

# 1. Load the secret API key from the .env file
load_dotenv()

app = Flask(__name__, static_url_path='', static_folder='.')

# 2. Configure Gemini API securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in .env file!")

TRANSLATIONS_FILE = 'translations.json'

def load_translations():
    if os.path.exists(TRANSLATIONS_FILE):
        try:
            with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_translations(data):
    with open(TRANSLATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Serve your HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Serve images and other assets
@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

# 3. Secure Translation API Endpoint
@app.route('/api/translate', methods=['POST'])
def translate_api():
    if not GEMINI_API_KEY:
        return jsonify({"error": "API key missing in .env"}), 500
        
    data = request.json
    lang_code = data.get('langCode')
    target_language = data.get('targetLanguage')
    keys_object = data.get('keysObject', {})

    print(f"\n🌍 Translation requested for: {target_language} ({lang_code})")

    # Load existing translations from local JSON file
    cache = load_translations()
    if lang_code not in cache:
        cache[lang_code] = {}

    cached_lang = cache[lang_code]
    missing_keys = {}

    # Find which words are not translated yet in the JSON file
    for key in keys_object.keys():
        if key not in cached_lang:
            missing_keys[key] = key

    # If there are missing words, translate ONLY the missing words using Gemini API
    if missing_keys:
        print(f"🔍 Found {len(missing_keys)} missing words. Translating via Gemini API...")
        prompt = f"You are a strict JSON translation API. Translate the values of the following JSON object into {target_language}. KEEP THE KEYS EXACTLY IN ENGLISH. Return ONLY a valid JSON object. Do not include markdown formatting.\n\n{json.dumps(missing_keys)}"

        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Force the model to output strict JSON to prevent Decode Errors
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Clean up the text to ensure it is valid JSON (removing any accidental markdown blocks)
            text_response = response.text.strip()
            text_response = re.sub(r"^```json\s*", "", text_response, flags=re.IGNORECASE)
            text_response = re.sub(r"^```\s*", "", text_response)
            text_response = re.sub(r"\s*```$", "", text_response)
                
            new_translations = json.loads(text_response.strip())

            # Merge the new translations into our cache
            for k, v in new_translations.items():
                cached_lang[k] = v

            # Save the updated translations back to translations.json
            save_translations(cache)
            print(f"✅ Successfully translated and saved {len(new_translations)} new words to translations.json!")
            
        except Exception as e:
            print("❌ Translation API Error:", e)
            # Proceed anyway, returning what we already have cached + English fallbacks
    else:
        print("⚡ All words are already in translations.json! Zero API calls made. Loading instantly.")

    # Build the final response using our fully updated cache
    final_response = {}
    for key in keys_object.keys():
        final_response[key] = cached_lang.get(key, key) # Fallback to English if completely missing

    return jsonify(final_response)

# 4. Chatbot Assistant Endpoint
@app.route('/chat', methods=['POST'])
def chat_api():
    if not GEMINI_API_KEY:
        return jsonify({"reply": "System Error: API Key missing in .env"}), 500
        
    data = request.json
    user_message = data.get('message', '')
    language = data.get('language', 'en')
    
    sys_prompt = f"You are the VaaniAI Health Assistant. You help users understand how VaaniAI's AI health companion works — covering voice emotion detection, habit tracking, sleep monitoring, weekly health scores, and emergency distress alerts. Keep answers under 2 sentences. Reply natively in the language code: {language}"
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=sys_prompt)
        response = model.generate_content(user_message)
        return jsonify({"reply": response.text})
    except Exception as e:
        print("Chat API Error:", e)
        return jsonify({"reply": "Sorry, I am having trouble connecting to the neural network."}), 500

if __name__ == '__main__':
    print("🚀 VaaniAI Backend is running!")
    print("🌐 Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)