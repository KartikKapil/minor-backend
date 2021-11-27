from django.db import models
from django.contrib.auth.models import User

class Notes(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    Title = models.CharField(max_length=100)
    Date_of_entry = models.DateField(auto_now_add=True)
    Entry = models.TextField()
    Emotion = models.CharField(max_length=15)
