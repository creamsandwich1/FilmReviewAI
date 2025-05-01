from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string


# Create your views here.
def model_view(request):
    return render(request, 'sentimentmodel/model_page.html')


def home_page_view(request):
    return render(request, 'sentimentmodel/main_page.html')
