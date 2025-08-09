# coffee/views.py
import os
import numpy as np
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
from django.conf import settings

# Import ALL necessary functions from disease_utils
from .disease_utils import (  # Remove get_openai_disease_details import
    preprocess_image, 
    classify_infection_level,
    get_disease_details,
    DISEASE_LABELS,
    get_openai_disease_details
)

logger = logging.getLogger(__name__)

class PredictDiseaseView(APIView):
    parser_classes = [MultiPartParser]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interpreter = self.load_model()
        
    def load_model(self):
        model_path = os.path.join(settings.BASE_DIR, 'coffee', 'tflite_model', 'coffee_model.tflite')
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def post(self, request):
        image = request.FILES.get('image')
        if not image:
            return Response({'error': 'No image uploaded'}, status=400)
        
        try:
            img_array = preprocess_image(image)
            
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            self.interpreter.set_tensor(input_details[0]['index'], img_array)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            
            prediction_idx = np.argmax(output_data)
            prediction_confidence = np.max(output_data)
            predicted_disease = DISEASE_LABELS[prediction_idx]
            
            infected = predicted_disease != "Healthy"
            infection_level = classify_infection_level(prediction_confidence)
            
            details = get_disease_details(
                predicted_disease, 
                prediction_confidence,
                infection_level
            )
            
            return Response({
                "predicted_disease": predicted_disease,
                "confidence": float(round(prediction_confidence, 4)),
                "affected": infected,
                "infection_level": infection_level,
                "details": details
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)


