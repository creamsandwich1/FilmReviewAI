from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from .bert.bert_model import classificate


# Create your views here.
def model_view(request):
    # 'Such a good film. Brilliant actors. Beautifull locations'
    # "That was awfull. Horrible cast. Lousy scenario. Don't waste your time."
    data = {
        'mark': classificate("That was awfull. Horrible cast. Lousy scenario. Don't waste your time.")
    }
    return render(request, 'sentimentmodel/model_page.html', context=data)


def home_page_view(request):
    return render(request, 'sentimentmodel/main_page.html')
