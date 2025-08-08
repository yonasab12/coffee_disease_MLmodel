from django.urls import path
from .views import PredictDiseaseView



urlpatterns = [ 
    path('predict', PredictDiseaseView.as_view(), name='predict_crop'),
    
]