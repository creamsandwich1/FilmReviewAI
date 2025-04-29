import re
import nltk
import contractions
import inflect

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup


def text_preprocessing(text: str) -> str:
    punkts = ['/', '|', '\\', '<', '>', '`', '^', '_', '(', ')', '{', '}',
              '[', ']', '*', '"', ',', '.', '?', '!', '+', '-', ':', ';', '@', "#", '%']

    # remove html tags and web links
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    cleaned_text = re.sub(r'https\S', '', cleaned_text)

    # remove punctuation
    pattern = '|'.join(re.escape(word) for word in punkts)
    cleaned_text = re.sub(pattern, '', cleaned_text)

    # replace &amp; with & and & with "and"
    cleaned_text = re.sub(r'&amp;', '&', cleaned_text)
    cleaned_text = re.sub(r'&', 'and', cleaned_text)

    # replace $ with dollar
    cleaned_text = re.sub(r'\$', 'dollar', cleaned_text)

    # replace non ascii symbols with space
    cleaned_text = re.sub(r'[^\x00-\x7F]', ' ', cleaned_text)

    # remove trailing spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # make lowercase
    cleaned_text = cleaned_text.lower()

    # fixes contractions such as `you're` to you `are`
    cleaned_text = contractions.fix(cleaned_text)

    # replace nums with words
    cleaned_text = number_to_text(cleaned_text)

    return cleaned_text


def number_to_text(text: str) -> str:
    temp_str = text.split()
    string = []
    for word in temp_str:
        # if the word is digit, converted to word else the sequence continues
        if word.isdigit():
            temp = inflect.engine().number_to_words(word)
            string.append(temp)
        else:
            string.append(word)

    return " ".join(string)


def tokenize(text: str) -> list:
    tokens = nltk.word_tokenize(text)
    return tokens


def remove_stopwords(lst: list) -> list:
    clean = []
    for word in lst:
        if word not in stopwords.words('english'):
            clean.append(word)
    return clean


def lemmatization(lst: list) -> list:
    lemma = WordNetLemmatizer()
    lemmas = []
    for i in lst:
        lem = lemma.lemmatize(i, pos='v')
        lemmas.append(lem)
    return lemmas


def join_text(lst: list) -> str:
    return " ".join(lst)


def normalize_text(text: str):
    process_funcs = [text_preprocessing, tokenize, remove_stopwords, lemmatization, join_text]
    for func in process_funcs:
        text = func(text)
    return text


def download_nltk() -> None:
    # download for the first time
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
