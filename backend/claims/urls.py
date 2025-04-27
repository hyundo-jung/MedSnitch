from django.urls import path
from .views import submit_claim_data

urlpatterns = [
    path('submit/', submit_claim_data, name='submit_claim_data'),
]
