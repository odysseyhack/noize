from django.db import models
from django.urls import reverse
from datetime import datetime


class Industry(models.Model):
    name = models.CharField(max_length=256)
    owner = models.CharField(max_length=256)
    location = models.CharField(max_length=256)
    #disaster = models.ForeignKey(Disaster,related_name='industries',on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("demo:detail",kwargs={'pk':self.pk})

class Disaster(models.Model):
    name = models.CharField(max_length=256)
    # each disaster will be assigned to an industry
    industry = models.ForeignKey(Industry,related_name='industries',on_delete=models.CASCADE)
    date = models.DateTimeField(default=datetime.now, blank=True)

    def __str__(self):
        return self.name
