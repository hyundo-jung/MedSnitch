from django.urls import path
from . import views

urlpatterns = [
    path('submit/', views.submit_claim_data, name='submit_claim_data'),
    path('users/', views.list_user_claims, name='user_claims'),
]
