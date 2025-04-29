import os
import torch

from django.conf import settings
from transformers import BertForSequenceClassification
from text_preprocessing import normalize_text

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights_path = os.path.join(settings.BASE_DIR, 'filmreviews', 'sentimentmodel',
                            'bert_classificator', 'bert_weights', '2', 'bert_st_dict_2.pt')
model = BertForSequenceClassification.from_pretrained(weights_path, num_labels=2).to(DEVICE)


def classificate(text):
    prepared_seq = normalize_text(text)
    output = model(prepared_seq)
    mark = torch.argmax(output).item()
    return mark
