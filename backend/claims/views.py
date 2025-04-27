from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import MedicalClaim
import numpy as np
import pandas as pd
import json
import torch
from .model import get_model_handler


def submit_claim_data(request):
    if request.method == 'POST':
        claim = MedicalClaim.objects.create(
            user=request.user if request.user.is_authenticated else None,
            claimType=request.POST.get('claimType'),
            StayDuration=request.POST.get('StayDuration'),
            cost=request.POST.get('cost'),
            num_diagnoses=request.POST.get('num_diagnoses'),
            DiagnosisCategory=request.POST.get('DiagnosisCategory'),
            num_procedures=request.POST.get('num_procedures'),
            first_procedure=request.POST.get('first_procedure'),
            Gender=request.POST.get('Gender'),
            Race=request.POST.get('Race'),
            isWeekend=request.POST.get('isWeekend'),
            ClaimDuration=request.POST.get('ClaimDuration'),
            ClaimDate=request.POST.get('ClaimDate'),
            Age=request.POST.get('Age'),
            first_diagnosis=request.POST.get('first_diagnosis')
        )

        claim_date = pd.to_datetime(request.POST.get('ClaimDate'))
        claim_day_of_year = claim_date.dayofyear

        input_features = [
            float(request.POST.get('claimType')),
            float(request.POST.get('StayDuration')),
            float(request.POST.get('cost')),
            float(request.POST.get('num_diagnoses')),
            float(request.POST.get('DiagnosisCategory')),
            float(request.POST.get('num_procedures')),
            float(request.POST.get('first_procedure')),
            float(request.POST.get('Gender')),
            float(request.POST.get('Race')),
            float(request.POST.get('isWeekend')),
            float(request.POST.get('ClaimDuration')),
            claim_day_of_year, # last three features are not scaled
            float(request.POST.get('Age')),
            float(request.POST.get('first_diagnosis'))
        ]

        model_handler = get_model_handler()

        features_array = np.array(input_features[:11]).reshape(1, -1)

        features_df = pd.DataFrame(features_array, columns=[
            'claimType', 'StayDuration', 'cost', 'num_diagnoses',
            'DiagnosisCategory', 'num_procedures', 'first_procedure',
            'Gender', 'Race', 'isWeekend', 'ClaimDuration'
        ])

        scaled_array = model_handler.scaler.transform(features_df)

        complete_input = np.hstack([scaled_array, np.array(input_features[11:]).reshape(1, -1)])

        nn_result = model_handler.predict_nn(torch.tensor(complete_input, dtype=torch.float32))
        xgb_result = model_handler.predict_xgb(complete_input)

        with torch.no_grad():
            raw_nn_output = torch.sigmoid(model_handler.nn_model(torch.tensor(complete_input, dtype=torch.float32))).item()

        xgb_probabilities = model_handler.xgb_model.predict_proba(complete_input)[0]
        raw_xgb_output = xgb_probabilities[1]

        
        response_data = {
            'nn_prediction': raw_nn_output,
            'xgb_prediction': raw_xgb_output,
            'nn_label': 'Fraudulent' if raw_nn_output > 0.5 else 'Legitimate',
            'xgb_label': 'Fraudulent' if raw_xgb_output > 0.5 else 'Legitimate',
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Only POST method allowed'}, status=400)

    #     return render(request, 'claims/submit_success.html', {
    #         'claim_id': claim.id,
    #         'nn_prediction': f"{raw_nn_output:.4f} ({'Fraudulent' if nn_result else 'Legitimate'})",
    #         'xgb_prediction': f"{raw_xgb_output:.4f} ({'Fraudulent' if xgb_result else 'Legitimate'})"
    #     })

    # else:
    #     # Render a form for GET request
    #     return render(request, 'claims/submit_claim.html')