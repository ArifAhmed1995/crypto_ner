"""Use methods in scrape_tools to load data into the KeyphraseExtractor
and generate dataset of crypto related keyphrase/words.
"""
import pickle
import pandas as pd

from scrape_tools import get_data, get_messari_news, clean_article, get_abbs_terms
from nlp_tools import keyphrase_extractor


def get_telegram_data():
    '''Load telegram chat data
    '''
    return list(pd.read_csv(r'./datasets/data.csv')['content'].dropna())


def get_leaf_sites(source):
    '''Get all relevant hyperlinked sites from a webpage
    which lead to crypto-related articles.
    '''
    leaf_sites = []

    if source == 'thetie':
        # Research TheTie
        home_site = 'https://research.thetie.io/category/research/'
        sites = [home_site + 'page/' + str(i) for i in range(1, 11)]
        #sites = [home_site + 'page/' + str(i) for i in range(1, 3)]

        leaf_sites = [leaf_site for site in sites for leaf_site in get_data(
            site, 'links', 'thetie')]
    elif source == 'block':
        # The Block
        home_site = 'https://www.theblock.co/category/defi?query=matchall&start='
        sites = [home_site+str(i) for i in range(0, 410, 10)]
        #sites = [home_site+str(i) for i in range(0, 20, 10)]

        for site in sites:
            for leaf_site in get_data(site, 'links', 'block'):
                if leaf_site not in leaf_sites:
                    leaf_sites.append(leaf_site)

    return leaf_sites


def filter_phrases(lst):
    '''Keyphrases extracted from the text data are filtered.
    Insignificant keyphrases are discarded
    '''
    res = []
    for i in range(len(lst)):
        split_words = lst[i].split(" ")
        # For larger keyphrases remove stray letters
        # Usually they are left over from the regex pipeline
        if len(split_words) > 2:
            res.extend([word for word in split_words if len(word) > 1])
        else:
            if len(lst[i]) > 1:
                res.append(lst[i])
    return res


def get_keyphrases(source):
    '''Major function to generate the list of keyphrases.
    Gets the leaf sites from a particular domain and processes
    each of those articles to expand crypto-related vocabulary
    '''
    keyphrases = list()
    extractor = keyphrase_extractor()

    if source in ['thetie', 'block']:
        sites = get_leaf_sites(source)
        article_count = 0
        five_art = ""
        for site in sites:
            if article_count >= 5:
                article_count = 1
                keyphrases.extend(filter_phrases(extractor(five_art)))
                five_art = get_data(site, 'text', source)
            else:
                article_count += 1
                five_art += " " + get_data(site, 'text', source)
    elif source == 'messari':
        news_data = get_messari_news()
        article_count = 0
        ten_art = ""
        for article in news_data:
            if article_count >= 10:
                article_count = 1
                keyphrases.extend(filter_phrases(extractor(ten_art)))
                ten_art = article
            else:
                article_count += 1
                ten_art += " " + article

    return list(set(keyphrases))


def get_and_save_sentences():
    '''Similar to keyphrase/word extraction but here sentences are
    extracted and stored instead. This is necessary to train a semantic
    model which would provide a feature representation for both noun phrases
    as well as extracted keyphrases/words.
    '''
    sentences = list()

    for source in ['thetie', 'block']:
        sites = get_leaf_sites(source)
        for site in sites:
            print("Processing site --> ", site)
            sentences.extend(get_data(site, 'article', source).split("."))

    news_data = get_messari_news()
    for article in news_data:
        print("Processing article --> ", article[0:100])
        sentences.extend(clean_article(article).split("."))

    # Save sentence corpus
    with open('./datasets/crypto_sentences.pkl', 'wb') as file:
        pickle.dump(sentences, file)

    return sentences


def augment_vocabulary():
    '''Call helper methods to build and expand vocabulary
    '''
    keyphrases = []

    abbs, terms = get_abbs_terms()
    keyphrases.extend(abbs)
    keyphrases.extend(terms)
    keyphrases.extend(get_keyphrases('thetie'))
    print("TheTie: Got keyphrases..............")
    keyphrases.extend(get_keyphrases('block'))
    print("The Block: Got keyphrases...........")
    keyphrases.extend(get_keyphrases('messari'))
    print("Messari IO: Got keyphrases..........")

    keyphrases = list(set(keyphrases))

    with open('./datasets/keyphrases.pkl', 'wb') as file:
        pickle.dump(keyphrases, file)
