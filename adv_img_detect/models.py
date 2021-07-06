from django.db import models

# Create your models here.
class ImageUpload(models.Model):
    # name = models.CharField(max_length=255, default='inputimage.png')
    pic = models.ImageField(default='default.png',upload_to = "images/", blank= True, null = True)
    def __str__(self):
        return self.pic


# class Result(models.Model):
#     cell_condition = models.CharField(max_length=100)
