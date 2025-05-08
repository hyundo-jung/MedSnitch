from django.db import models
from django.contrib.auth.models import User

class MedicalClaim(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    claimType = models.IntegerField()
    StayDuration = models.FloatField()
    cost = models.FloatField()
    num_diagnoses = models.IntegerField()
    DiagnosisCategory = models.IntegerField()
    num_procedures = models.IntegerField()
    first_procedure = models.IntegerField()
    Gender = models.IntegerField()
    Race = models.IntegerField()
    ClaimDuration = models.FloatField()
    ClaimDate = models.DateField(null=True, blank=True)
    Age = models.IntegerField()

    isWeekend = models.BooleanField(default=False)
    ClaimdDy_Sin = models.FloatField()
    ClaimDay_Cos = models.FloatField()


    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Claim {self.id} by {self.user if self.user else 'Anonymous'}"
