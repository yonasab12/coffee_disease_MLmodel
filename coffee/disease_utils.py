# coffee/disease_utils.py
import re
import logging
import numpy as np
from PIL import Image
import openai
from django.conf import settings
from .constants import DISEASE_LABELS, OPENAI_PROMPT_TEMPLATE

# Configure logger
logger = logging.getLogger(__name__)

def classify_infection_level(confidence):
    if confidence < 0.5:
        return "Not Affected"
    elif confidence < 0.7:
        return "Mild"
    elif confidence < 0.9:
        return "Moderate"
    return "Severe"

def preprocess_image(image):
    img = Image.open(image).resize((224, 224)).convert('RGB')
    img = np.array(img) / 255.0
    return img.astype(np.float32).reshape(1, 224, 224, 3)

def get_openai_disease_details(disease_name, confidence, infection_level):
    try:
        logger.debug(f"Calling OpenAI API for {disease_name}")
        logger.debug(f"USE_OPENAI: {settings.USE_OPENAI}")
        logger.debug(f"API Key: {settings.OPENAI_API_KEY[:5]}...")
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        infection_level_lower = infection_level.lower()
        
        prompt = OPENAI_PROMPT_TEMPLATE.format(
            disease=disease_name,
            confidence=round(confidence*100, 1),
            infection_level=infection_level,
            infection_level_lower=infection_level_lower
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're an agricultural expert specializing in coffee plant diseases."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI response received: {content[:200]}...")
        return parse_openai_response(content)
    
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return None

def parse_openai_response(content):
    """Robust parsing with fallback"""
    sections = {
        "summary": r"(?i)summary[:\s-]*(.*?)(?=(?:organic|chemical|environmental|$))",
        "organic_treatment": r"(?i)organic treatment[:\s-]*(.*?)(?=(?:chemical|environmental|$))",
        "chemical_treatment": r"(?i)chemical treatment[:\s-]*(.*?)(?=(?:environmental|$))",
        "environmental_advice": r"(?i)environmental advice[:\s-]*(.*)"
    }
    
    result = {}
    for key, pattern in sections.items():
        try:
            match = re.search(pattern, content, re.DOTALL)
            result[key] = match.group(1).strip() if match else ""
        except Exception:
            logger.warning(f"Failed to parse section: {key}")
            result[key] = ""
    
    if not any(result.values()):
        logger.error(f"Failed to parse all sections from OpenAI response:\n{content[:500]}")
        return None
    
    return result

# Static disease information remains the same ...
# Static disease information
STATIC_DISEASE_INFO = {
    "Healthy": {
        "summary": "The coffee plant shows no signs of disease",
        "organic_treatment": "Maintain proper nutrition and watering",
        "chemical_treatment": "Regular monitoring",
        "environmental_advice": "Ensure adequate shade and proper drainage"
    },
    "Coffee Rust": {
        "summary": "Fungal disease causing orange powdery spots on leaves",
        "organic_treatment": "Apply copper-based fungicides or baking soda solution",
        "chemical_treatment": "Triazole fungicides (e.g., Triademefon)",
        "environmental_advice": "Improve air circulation, reduce leaf wetness duration"
    },
    "Leaf Miner": {
        "summary": "Larvae tunnel through leaves creating serpentine mines",
        "organic_treatment": "Neem oil or spinosad applications",
        "chemical_treatment": "Abamectin or Cyromazine insecticides",
        "environmental_advice": "Remove infected leaves, encourage natural predators"
    },
    "Phoma": {
        "summary": "Fungal disease causing dark lesions and leaf drop",
        "organic_treatment": "Copper hydroxide sprays",
        "chemical_treatment": "Triazole fungicides",
        "environmental_advice": "Avoid overhead irrigation, improve soil drainage"
    },
    "Cercospora": {
        "summary": "Causes brown spots with yellow halos on leaves",
        "organic_treatment": "Sulfur-based fungicides",
        "chemical_treatment": "Chlorothalonil or Mancozeb",
        "environmental_advice": "Space plants properly, remove infected plant material"
    }
}
def get_disease_details(disease_name, confidence, infection_level):
    logger.debug(f"USE_OPENAI setting: {getattr(settings, 'USE_OPENAI', 'not set')}")
    logger.debug(f"OpenAI key present: {bool(getattr(settings, 'OPENAI_API_KEY', False))}")
    # Use settings.USE_OPENAI directly
    if disease_name != "Healthy" and settings.USE_OPENAI:
        logger.info(f"Fetching OpenAI details for {disease_name}")
        openai_details = get_openai_disease_details(
            disease_name, confidence, infection_level
        )
        
        if openai_details:
            logger.info("Successfully retrieved details from OpenAI")
            return {**openai_details, "source": "openai"}
        else:
            logger.warning("OpenAI details failed, falling back to static data")
    
    # Fallback to static data
    static_info = STATIC_DISEASE_INFO.get(
        disease_name, 
        STATIC_DISEASE_INFO['Healthy']
    )
    return {**static_info, "source": "static"}