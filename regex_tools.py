"""Regex Methods for pre-processing different types of textual content(articles, messages, scraped text)

Most of these helper regex methods were directly taken from StackOverflow and bunch
of NLP websites. Some simpler ones were written from scratch.
"""

import re
import spacy

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import ToktokTokenizer


def remove_links(st):
    '''Removes any type of links 
    '''
    return re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', st)


def remove_emojis(data):
    '''Removes emojis and other non-English special characters
    '''
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def remove_non_core(text):
    '''Keeps only strict core english components which are alphabets, ' and -
    Removes numbers too as this might interfere with accurate similarity score.
    '''
    return re.sub(r'[^A-Za-z\'-]+', ' ', text)


def remove_non_core_sentence(text):
    '''Removes , as well from articles where there is a space after comma
    '''
    return re.sub(r'[^A-Za-z\'-.]+', ' ', re.sub(r',', '', text))


def replace_new_line_with_space(text):
    '''Replaces new line character with space
    '''
    return re.sub(r'\n', ' ', text)


def remove_tags(text):
    '''Replace all HTML tags and content within inside them with space
    '''
    return re.sub(re.compile('<.*?>'), ' ', text)


def remove_extra_spaces(text):
    '''Replaces consecutive spaces with only one space
    '''
    return " ".join(text.split())


def remove_spaces(text):
    '''Remove spaces from the given text
    '''
    return "".join(text.split())


def remove_unnecessary_words(text):
    '''Removes stopwords and stray single letters
    '''
    en = spacy.load('en_core_web_sm')
    return ' '.join([word for word in text.split() if word.lower() not in en.Defaults.stop_words and len(word) > 1])


def lemmatize_text(text):
    '''Lemmatizes the words in the text. Each word is reduced to a meaningful base form.
    '''
    lem = WordNetLemmatizer()
    return lem.lemmatize(text)


def remove_numbers(text):
    '''Removes all types of numbers except ones which are linked
    with a word(e.g '3.33 blockchain' --> 3.33 is removed, 'web3' --> 3 is NOT removed)
    '''
    return re.sub(r'^\d+\s|\s\d+\s|\s\d+$', '', re.sub(r'(?<=\d)[,\.]', '', text))


def remove_adj_adv(text):
    '''Removes adverbs(all types) and adjectives(comparative, superlative) from the text.
    Removing absolute form adjectives can omit some important keywords, hence they're preserved.
    '''
    token = ToktokTokenizer()
    # JJ -> Adjective, RB -> Adverb, R -> Comparative, S -> Superlative
    adv_adjective_tag_list = set(['JJR', 'JJS', 'RB', 'RBR', 'RBS'])
    words = token.tokenize(text)

    # Tag the words with their part of speech
    words_tagged = pos_tag(tokens=words, tagset=None, lang='eng')

    # Filter out words which are part of the tag list
    filtered = [w[0]
                for w in words_tagged if w[1] not in adv_adjective_tag_list]

    return ' '.join(map(str, filtered))


def clean_scraped_text(text_data):
    '''Pre-processing pipeline for scraped text data from websites
    '''
    return lemmatize_text(remove_extra_spaces(remove_non_core(remove_unnecessary_words(remove_links(remove_tags(text_data.lower()))))))


def clean_message(message):
    '''Pre-processing pipeline for telegram messages. Should work on messages from other platforms too.
    '''
    return remove_extra_spaces(remove_links(remove_numbers(remove_tags(remove_emojis(message)))))


def clean_article(text_data):
    '''Pre-processing pipeline for news articles.
    '''
    return remove_unnecessary_words(lemmatize_text(remove_extra_spaces(remove_non_core_sentence(remove_unnecessary_words(remove_links(remove_tags(text_data.lower())))))))


if __name__ == '__main__':
    text = '''
    Processing raw text intelligently is difficult: most words are rare, \
    and it\'s common for words that look completely different to mean almost the same thing.\
    The same words in a different order can mean something completely different. Even splitting text\
    into useful word-like units can be difficult in many languages. While it\'s possible to solve some problems\
    starting from only the raw characters, it\'s usually better to use linguistic knowledge to add useful information.\
    That\'s exactly what spaCy is designed to do: you put in raw text, and get back a Doc object, that comes with a variety of annotations.
    '''
    print(clean_article(text).split("."))
