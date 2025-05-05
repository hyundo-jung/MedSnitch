from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
import json

from django.views.decorators.csrf import csrf_exempt

def home(request):
    if request.user.is_authenticated:
        return JsonResponse({'message': f"Hello, {request.user.username}!"})
    else:
        return JsonResponse({'error': 'User not authenticated'}, status=401)

# def home(request):
#     return HttpResponse(f"Hello, {request.user.username}!")
    
@csrf_exempt
def register(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Username already exists.'}, status=400)
        else:
            User.objects.create_user(username=username, password=password)
            return JsonResponse({'message': 'User registered successfully.'})

    return JsonResponse({'error': 'Only POST method allowed.'}, status=405)

   
# def register(request):
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']
#         if User.objects.filter(username=username).exists():
#             messages.error(request, 'Username already exists.')
#         else:
#             User.objects.create_user(username=username, password=password)
#             return redirect('login')
#     return render(request, 'accounts/register.html')

import json

def login_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({'message': 'Login successful', 'username': user.username})
        else:
            return JsonResponse({'error': 'Invalid credentials.'}, status=401)

    return JsonResponse({'error': 'Only POST method allowed.'}, status=405)


# def login_view(request):
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']
#         user = authenticate(request, username=username, password=password)
#         if user is not None:
#             login(request, user)
#             return redirect('home')  # Replace with your main page
#         else:
#             messages.error(request, 'Invalid credentials.')
#     return render(request, 'accounts/login.html')

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return JsonResponse({'message': 'Logged out successfully.'})
    return JsonResponse({'error': 'Only POST method allowed.'}, status=405)

# def logout_view(request):
#     logout(request)
#     return redirect('login')
