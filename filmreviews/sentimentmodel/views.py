from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from .bert.bert_model import classificate
from .bert.text_preprocessing import detect_ru_and_translate


# Create your views here.
def model_view(request):
    # 'Such a good film. Brilliant actors. Beautifull locations.'
    # "That was awfull. Horrible cast. Lousy scenario. Don't waste your time."
    if request.method == 'POST':
        input_text = request.POST.get('text', '')
        input_text = detect_ru_and_translate(input_text)
        predictions = classificate(input_text)
        label = 'Положительный' if predictions == 1 else 'Отрицательный'

        return JsonResponse({'result': label})
    return render(request, 'sentimentmodel/model_page.html')


def home_page_view(request):
    return render(request, 'sentimentmodel/main_page.html')
