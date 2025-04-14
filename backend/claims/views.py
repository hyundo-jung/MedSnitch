from django.shortcuts import render
from django.http import HttpResponse
from .models import MedicalClaim

def submit_claim_data(request):
    if request.method == 'POST':
        try:
            claim = MedicalClaim.objects.create(
                user=request.user if request.user.is_authenticated else None,
                claim_amount=request.POST.get('claim_amount'),
                claim_date=request.POST.get('claim_date'),
                procedure_code=request.POST.get('procedure_code'),
                patient_age=request.POST.get('patient_age'),
                patient_gender=request.POST.get('patient_gender'),
                provider_speciality=request.POST.get('provider_speciality'),
                claim_status=request.POST.get('claim_status'),
                patient_income=request.POST.get('patient_income'),
                patient_marital_status=request.POST.get('patient_marital_status'),
                patient_employment_status=request.POST.get('patient_employment_status'),
                claim_type=request.POST.get('claim_type'),
                submission_method=request.POST.get('submission_method')
            )

            return render(request, 'claims/submit_success.html', {'claim_id': claim.id})

        except Exception as e:
            return HttpResponse(f"Error: {str(e)}", status=400)

    return render(request, 'claims/submit_form.html')
