from django.db import models

class Document(models.Model):
	IMAGE_CLASSES = (
		(-1, 'Unknown'),
        (0, 'Image'),
        (1, 'Photo'),
        (2, 'Screenshot'),
    )
	image_file = models.FileField(upload_to='documents/%Y/%m/%d')
	image_class = models.IntegerField(default=-1, choices=IMAGE_CLASSES)