from rest_framework import serializers 


from .models import InteractiveModels 

class InteractiveSerializer(serializers.ModelSerializer):
    class Meta:

        model = InteractiveModels
        fields = ["companycode"]
