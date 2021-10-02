from django.db import models
from django.contrib.auth.models import User

class Notes(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    Date_of_entry = models.DateField(auto_now_add=True)
    Entry = models.TextField()
