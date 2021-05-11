from django.db import models

# Create your models here.



class InteractiveModels(models.Model):

    companycode = models.CharField(max_length = 256)

    def __str__(self):
        return self.companycode 



