import os
import torch
import gdown

from django.conf import settings
from transformers import BertTokenizer, BertForSequenceClassification
from sentimentmodel.bert.text_preprocessing import normalize_text

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

weights_path = os.path.join(settings.BASE_DIR, 'sentimentmodel',
                            'bert', 'weights', '3', 'bsd1.pt')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(DEVICE)

if not os.path.exists(weights_path):
    url = 'https://drive.google.com/file/d/1fcMJjIhsAS_jQfviilofvepItxyHQcdb/view?usp=drive_link'
    gdown.download(url, weights_path, quiet=False, fuzzy=True)

st = torch.load(weights_path, weights_only=False)
model.load_state_dict(state_dict=st)
model.eval()


def classificate(text):
    prepared_seq = normalize_text(text)

    encoded_dict = tokenizer.encode_plus(
        prepared_seq,  # текст строка которую кодируем
        add_special_tokens=True,  # добавить '[CLS]' и '[SEP]' токены
        max_length=512,  # параметр максимальной длины текста
        padding='max_length',  # делать падинг до макс длины
        truncation=True,  # если длина больше max_length то отрезать лишнее
        return_attention_mask=True,  # делать ли маску внимания
        return_tensors='pt',  # формат возвращаемого тензора
        # return_token_type_ids=False,
    )
    input_ids = encoded_dict['input_ids'].to(DEVICE)
    attention_mask = encoded_dict['attention_mask'].to(DEVICE)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    mark = torch.argmax(outputs.logits).item()
    return mark
