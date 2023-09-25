from django.db import models

# Create your models here.
class Pictures(models.Model):
#	#image = models.ImageField(null = True, blank = True,  upload_to = "images/")
	image = models.ImageField(null = True, blank = True, upload_to = "media/segmented_50000_v4/")
