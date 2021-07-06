from django.shortcuts import render,redirect

from django.http import HttpResponse
# from rest_framework.response import Response
from .models import ImageUpload
from PIL import Image

from .forms import *

def handler404(request,exception):
    # response = render(template_name)
    # response.status_code = 404
    # return response
    return render(request,'pages/404.html')

def handler500(request):
    # response = render(template_name)
    # response.status_code = 404
    # return response
    return render(request,'pages/500.html')

def home(request):
    
    if request.method == 'POST':
        
        form = ImageForm(request.POST, request.FILES)
        print(request.FILES)
        if form.is_valid():
            form.save()
            
            uploadedimage = ImageUpload.objects.latest('pic') 
            return render(request,'pages/index.html', {'form' : form,'has_image':True,'images' : uploadedimage})
    else:
        ImageUpload.objects.all().delete()
        form = ImageForm()
    return render(request,'pages/index.html', {'form' : form,'has_image':False})

def success(request):
    return HttpResponse('successfully uploaded')
