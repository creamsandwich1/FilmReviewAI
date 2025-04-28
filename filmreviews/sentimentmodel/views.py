from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.
def model_view(request):
    return HttpResponse('Страница модели')


def home_page_view(request):
    return HttpResponse('Главная страница')
