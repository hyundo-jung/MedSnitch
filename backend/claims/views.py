from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import MedicalClaim
import numpy as np
import pandas as pd
import json
import torch
from .model import get_model_handler
import os

from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@csrf_exempt
def submit_claim_data(request):
    print("Request method:", request.method)

    if request.method == 'POST':

        claim_date = pd.to_datetime(request.POST.get('ClaimDate'))
        claim_day_of_year = claim_date.dayofyear
        is_weekend = claim_date.weekday() >= 5
        claimday_sin = np.sin(2 * np.pi * claim_day_of_year / 365)
        claimday_cos = np.cos(2 * np.pi * claim_day_of_year / 365)

        input_features = [
            float(request.POST.get('StayDuration')),
            float(request.POST.get('cost')),
            float(request.POST.get('num_diagnoses')),
            float(request.POST.get('num_procedures')),
            float(request.POST.get('Gender')),
            float(request.POST.get('Race')),
            float(claimday_sin),
            float(claimday_cos),
            float(request.POST.get('ClaimDuration')),

            float(request.POST.get('claimType')),  # Not scaled
            float(request.POST.get('first_procedure')),
            float(request.POST.get('Age')),
            float(is_weekend),
            float(request.POST.get('DiagnosisCategory'))
        ]

        # Split into scaled and unscaled features
        scaled_features = input_features[:9]
        unscaled_features = input_features[9:]

        # Convert to numpy arrays
        scaled_array = np.array(scaled_features).reshape(1, -1)
        unscaled_array = np.array(unscaled_features).reshape(1, -1)

        # Corrected DataFrame column names
        scaled_df = pd.DataFrame(scaled_array, columns=[
            'StayDuration', 'cost', 'num_diagnoses', 'num_procedures',
            'Gender', 'Race', 'ClaimDay_sin', 'ClaimDay_cos', 'ClaimDuration'
        ])

        # Get the model handler with the correct number of diagnostic categories
        model_handler = get_model_handler()

        # Scale the scaled features
        scaled_array = model_handler.scaler.transform(scaled_df)

        mapped_diag_cat = model_handler.map_icd_to_ccs(request.POST.get('DiagnosisCategory'))

        # Combine scaled and unscaled features
        features_array = np.hstack([scaled_array, unscaled_array])

        x_numeric = torch.tensor(features_array[:, :-1], dtype=torch.float32)  # All features except the last one
        x_diag_cat = torch.tensor(mapped_diag_cat, dtype=torch.long)      # Diagnosis category as a separate tensor


        print("features_array shape:", features_array.shape)
        print("x_numeric shape:", x_numeric.shape)
        print("x_diag_cat:", x_diag_cat)

        if x_numeric.ndim == 1:
            x_numeric = x_numeric.unsqueeze(0)
        if x_diag_cat.ndim == 0:
            x_diag_cat = x_diag_cat.unsqueeze(0)

        nn_result = model_handler.predict_nn(x_numeric, x_diag_cat)
        xgb_result = model_handler.predict_xgb(features_array)

        with torch.no_grad():
            raw_nn_output = torch.sigmoid(model_handler.nn_model(x_numeric, x_diag_cat)).item()

        xgb_probabilities = model_handler.xgb_model.predict_proba(features_array)[0]
        raw_xgb_output = xgb_probabilities[1]

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
            ClaimDuration=request.POST.get('ClaimDuration'),
            ClaimDate=request.POST.get('ClaimDate'),
            Age=request.POST.get('Age'),
            isWeekend=is_weekend,
            ClaimdDy_Sin=claimday_sin,
            ClaimDay_Cos=claimday_cos,
            nn_prediction=raw_nn_output,
            xgb_prediction=raw_xgb_output
        )
        
        response_data = {
            'nn_prediction': float(raw_nn_output),
            'xgb_prediction': float(raw_xgb_output),
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
    


@login_required
def list_user_claims(request):
    user_claims = MedicalClaim.objects.filter(user=request.user).order_by('-created_at')

    claims_list = []
    for claim in user_claims:
        claims_list.append({
            'id': claim.id,
            'claim_type': claim.claimType,
            'stay_duration': claim.StayDuration,
            'cost': claim.cost,
            'num_diagnoses': claim.num_diagnoses,
            'diagnosis_category': claim.DiagnosisCategory,
            'num_procedures': claim.num_procedures,
            'first_procedure': claim.first_procedure,
            'gender': claim.Gender,
            'race': claim.Race,
            'claim_duration': claim.ClaimDuration,
            'claim_date': claim.ClaimDate.strftime('%Y-%m-%d'),
            'age': claim.Age,
            'is_weekend': claim.isWeekend,
            'nn_prediction': round(claim.nn_prediction, 2) if claim.nn_prediction is not None else None,
            'xgb_prediction': round(claim.xgb_prediction, 2) if claim.xgb_prediction is not None else None,
            'nn_label': 'Fraudulent' if (claim.nn_prediction if claim.nn_prediction is not None else 0) > 0.2464 else 'Legitimate',
            'xgb_label': 'Fradulent' if (claim.xgb_prediction if claim.xgb_prediction is not None else 0) > 0.5 else 'Legitimate'
        })

    return JsonResponse({'claims': claims_list}, safe=False)

    # return render(request, 'claims/user_claims.html', {
    #     'user_claims': user_claims
    # })
    