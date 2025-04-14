from django.db import models
from django.contrib.auth.models import User

class MedicalClaim(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    claim_amount = models.FloatField()
    claim_date = models.DateField()
    procedure_code = models.CharField(max_length=20)
    patient_age = models.IntegerField()
    patient_gender = models.CharField(max_length=10, choices=[("Male", "Male"), ("Female", "Female"), ("Other", "Other")])
    provider_speciality = models.CharField(max_length=100)
    claim_status = models.CharField(max_length=50)
    patient_income = models.FloatField()
    patient_marital_status = models.CharField(max_length=20, choices=[("Single", "Single"), ("Married", "Married"), ("Divorced", "Divorced")])
    patient_employment_status = models.CharField(max_length=20, choices=[("Employed", "Employed"), ("Unemployed", "Unemployed"), ("Retired", "Retired")])
    claim_type = models.CharField(max_length=50)
    submission_method = models.CharField(max_length=50)
    is_fraud = models.BooleanField(null=True, blank=True)  # ML result

    def __str__(self):
        return f"Claim #{self.id} by {self.user.username}"
