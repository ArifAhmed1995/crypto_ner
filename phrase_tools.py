"""Methods to extract noun phrases using different NLP libraries.
Any one by itself doesn't extract all relevant noun phrases.
"""


from textblob import TextBlob
from nltk import word_tokenize

from regex_tools import remove_unnecessary_words, remove_adj_adv, spacy, pos_tag


def clean_noun_phrases(nps):
    '''Remove adverbs + adjectives(comparative, superlative) from the noun phrase
    as well as stopwords and remaining stray single letters.
    '''
    return [remove_unnecessary_words(remove_adj_adv(noun_phrase)) for noun_phrase in nps]


def textblob_(sentence):
    '''Noun phrases from TextBlob
    '''
    text_blob = TextBlob(sentence)
    return text_blob.noun_phrases


def spacy_(model, sentence):
    '''Noun phrases from a SpaCy model
    '''
    return [str(nc) for nc in model(sentence).noun_chunks]


def nltk_(sentence):
    '''Noun phrases from NLTK
    '''
    tokens = word_tokenize(sentence)
    parts_of_speech = pos_tag(tokens)
    return [pos[0] for pos in parts_of_speech if pos[1] == 'NN']


def best_phrases(chat_log):
    '''Iterates over the chat log and generate the noun phrases
    for each message. Empty messages, > 4 words long phrases and
    <=2 letter phrases are not part of the result.

    This is the major bottleneck method since generating noun phrases
    requires a relatively large language model to be referred to during
    the process.

    TODO: Speed up this method. Multi-threading/processing does not
    work since overheads negate the benefit of multiple workers.
    '''
    res = []
    model = spacy.load('en_core_web_lg')
    for chat in chat_log:

        all_phrases = clean_noun_phrases(
            textblob_(chat) + spacy_(model, chat) + nltk_(chat))
        current_phrases = []

        # No empty phrases or substring repititions, > 4 word phrases
        # not allowed and phrase should be more than 2 chars
        for phrase in all_phrases:
            if phrase != '' and\
                not any(phrase in word for word in current_phrases) and\
                    len(phrase.split(' ')) < 4 and len(phrase) > 2:
                current_phrases.append(phrase)
        res.append(current_phrases)
    return res


if __name__ == '__main__':
    sts = ['i\'d argue reality is a downwards diagonal even, too kind with the bell curve',
           'We\'ve been doing it with just JavaScript and walletconnect']
    print(best_phrases(sts))
