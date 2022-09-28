"""Helper methods for NLP pipeline.
"""

import pickle

import numpy as np
import pandas as pd

from transformers.pipelines import AggregationStrategy

from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from gensim.models import Word2Vec


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(
                model, model_max_length=512),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])


def keyphrase_extractor():
    '''This model uses KBIR as its base model and fine-tunes it on the KPTimes dataset.

    Fine tuned on KPTimes, a large-scale dataset of news texts paired with
    editor-curated keyphrases. This is important since our keyphrase/word
    extraction is from news and blog articles.
    '''
    model_name = "ml6team/keyphrase-extraction-kbir-semeval2017"
    extractor = KeyphraseExtractionPipeline(model=model_name)

    return extractor


def lex_match_score(data, phrase):
    '''Returns lexicographical matching score of a phrase
    with respect to a vocabulary
    '''
    phrase = [word for word in phrase.lower().split(' ')]
    total_score = 0

    match_length = 0
    total_phrase_length = 0
    match_count = 0

    for word in phrase:
        total_phrase_length += len(word)
        for crypto_phrase in data:
            max_length = len(word) if len(word) > len(
                crypto_phrase) else len(crypto_phrase)
            # Check for substring match both ways. If yes,
            # increase total_score, match_count and match_length
            if word in crypto_phrase:
                total_score += len(word)/max_length
                match_length += len(word)
                match_count += 1
                break
            elif crypto_phrase in word:
                total_score += len(crypto_phrase)/max_length
                match_length += len(crypto_phrase)
                match_count += 1
                break
    match_score = total_score/match_count if match_count != 0 else 0
    match_percentage = match_length / \
        total_phrase_length if total_phrase_length != 0 else 0
    # The score is indicative of how much the given string matched with
    # a particular phrase in the vocabulary, hence the match_score
    # is normalized by match percentage.
    return match_score * match_percentage


def load_sentences():
    '''Load sentences acquired from crypto related articles
    '''
    data = None
    with open('./datasets/crypto_sentences.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


def get_keyphrases():
    '''Load previously acquired keyphrases
    '''
    data = None
    with open('./datasets/keyphrases.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


def generate_and_save_crypto_word2vec_model():
    '''Build a Word2Vec embedding from crypto related sentences
    '''
    data = load_sentences()

    # A high window value leads to better semantic matching.
    model = Word2Vec(sg=1, window=70, vector_size=50, workers=8)
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=200)
    model.save("./models/crypto.model")
