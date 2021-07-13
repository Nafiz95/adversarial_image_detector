from django.shortcuts import render,redirect

from django.http import HttpResponse
# from rest_framework.response import Response
from .models import ImageUpload
from .forms import *
from .imageProcessor import *
from pathlib import Path
import json
import pickle
# input_image_path = "/home/nafiz/Downloads/test_images/clean/AbdomenCT_2.png"
BASE_DIR = Path(__file__).resolve().parent.parent
with open(BASE_DIR / 'config.json') as config_file:
    config = json.load(config_file)
filename = config['MODEL_PATH']
loaded_model = pickle.load(open(filename, 'rb'))


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
        ImageUpload.objects.all().delete()
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            
            uploadedimage = ImageUpload.objects.latest('pic') 
            print(uploadedimage.pic.url)
            input_image_path = '.'+str(uploadedimage.pic.url)
            filter_dataframe = getFilterValues(input_image_path)
            pred = loaded_model.predict([filter_dataframe.loc[0]])
            
            return render(request,'pages/index.html', {'form' : form,'has_image':True,'pred':pred[0],'images' : uploadedimage})
    else:
        ImageUpload.objects.all().delete()
        form = ImageForm()
    return render(request,'pages/index.html', {'form' : form,'has_image':False})

def success(request):
    return HttpResponse('successfully uploaded')
