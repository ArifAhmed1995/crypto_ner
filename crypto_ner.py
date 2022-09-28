"""End user methods
"""

import heapq
import random

import spacy

from statistics import mean

from phrase_tools import best_phrases
from regex_tools import clean_message, remove_unnecessary_words
from vocab_tools import get_telegram_data
from nlp_tools import get_keyphrases, lex_match_score


def get_model_keyphrase_data():
    '''Loads and returns spacy model, keyphrases and their model embeddings.
    '''
    crypto_model = spacy.load('./models/spacy.crypto.word2vec.model')
    stopwords = spacy.load('en_core_web_sm').Defaults.stop_words
    keyphrase_data = get_keyphrases()
    return crypto_model, keyphrase_data, [crypto_model(str(kp)) for kp in keyphrase_data], stopwords


def get_keyphrase_matches(messages):
    '''Main end-user method to extract keywords given a list of messages
    '''
    crypto_model, keyphrase_data, model_embedded_keyphrases, stopwords = get_model_keyphrase_data()

    num_best_matches = 10
    keyphrase_matches = []
    for i, best_cleaned_phrases in enumerate(best_phrases(messages)):
        # Get the best noun-phrases
        crypto_phrases = []
        for phrase in best_cleaned_phrases:
            crypto_match_scores = []
            for kp in model_embedded_keyphrases:
                semantic_score = crypto_model(phrase.lower()).similarity(kp)
                # If there are less than num_best_matches keep pushing onto the heap
                if len(crypto_match_scores) < num_best_matches:
                    heapq.heappush(crypto_match_scores, semantic_score)
                # If heap is full and semantic score beats lowest score, pop lowest and push score onto heap
                elif semantic_score > heapq.nlargest(num_best_matches, crypto_match_scores)[-1]:
                    heapq.heappop(crypto_match_scores)
                    heapq.heappush(crypto_match_scores, semantic_score)

            # Compute mean semantic score and lexical match score
            semantic_net_score = mean(heapq.nlargest(10, crypto_match_scores))
            lms = lex_match_score(keyphrase_data, phrase)

            # Empirically determined constants
            computed_score = semantic_net_score * 0.65 + lms * 0.35

            if (computed_score > 0.6 and lms > 0.5) or semantic_net_score > 0.725:
                crypto_phrases.append(
                    (phrase, semantic_net_score, lms, computed_score))

        # Reverse compare with original message. This prevents reporting
        # phrases which might have lost a word in between due to pre or post processing
        crypto_phrases = [phrase[0]
                          for phrase in crypto_phrases if phrase[0] in messages[i] and phrase[0] not in stopwords]
        keyphrase_matches.append((messages[i], crypto_phrases))

    return keyphrase_matches


def test_random_messages(num_of_messages=10):
    '''Get a bunch of random messages from the telegram dataset
    and extract keywords
    '''
    tele_data = get_telegram_data()
    start_index = random.randint(0, len(tele_data) - num_of_messages)
    cleaned_chats = [clean_message(
        td) for td in tele_data[start_index:start_index + num_of_messages]]

    return get_keyphrase_matches(cleaned_chats)


def get_keyphrase_matches_single(message):
    '''Get crypto-related keyphrases/words for a single string
    '''
    return get_keyphrase_matches([clean_message(message)])[0]


if __name__ == '__main__':
    st1 = "This is super exciting. Using deep reinforcement learning to analyze Blockchain security and find even better selfish mining techniques"
    st2 = "By the way, which do you think is the on-chain analytics with the best user experience? Preferably the ones that are self-served and that everyone on the team can use"
    st3 = "We're launching incentivized testnet on polygon today at tokensoft"
    sts = [st1, st2, st3]

    print(get_keyphrase_matches(sts), '\n')
    print(get_keyphrase_matches_single(
        'simple public good stablecoin with ETH as backing fee accumulation as collateral'), '\n')

    for kms in test_random_messages(20):
        print(kms)
