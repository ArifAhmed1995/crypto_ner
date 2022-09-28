"""Scraping methods to acquire text data from Messari IO, The Block and TheTie.
"""
import requests

import pandas as pd

from bs4 import BeautifulSoup

from messari import Messari

from regex_tools import clean_scraped_text, clean_article


def get_abbs_terms():
    '''Add abbreviations and terms already procured.
    '''
    term_abb = pd.read_csv(r'./datasets/term_abb.csv').dropna()
    term_def = pd.read_csv(r'./datasets/term_def.csv').dropna()
    abbs = list(term_abb['abbreviations'])
    terms = list(term_def['terms']) + list(term_abb['abbreviations'])
    return abbs, terms


def get_messari_news():
    '''Use Messari API to get all news articles per page.
    Scrape the webpage content to get main article text
    '''
    messari = Messari('4a9b688a-59af-45e3-ac6c-7ae9b046dd83')

    messari_res = []
    page = 1
    while True:
        res = messari.get_all_news(page)
        # if 'data' not in res:
        if 'data' not in res or page == 50:
            break
        messari_res.extend([clean_scraped_text(article['content'])
                           for article in res['data']])
        page += 1
        print("Messari IO: Got data from page "+str(page))
    return messari_res


def text_from_html(response, source):
    '''Given the HTML of a webpage from TheTie or The Block, extract
    the main article text. Leave out Podcasts and Webinar since they contain
    minimal data.
    '''
    soup = BeautifulSoup(response.text, 'lxml')
    if source == 'thetie':
        title = soup.find('title').text
        text_data = ''
        if not any(substring in title for substring in ["[Podcast]", "[Webinar Recording]"]):
            text_data = soup.findAll(
                'div', class_=lambda x: x == 'entry-content')[0].text
        return str(text_data)
    elif source == 'block':
        text_data = soup.findAll(
            'div', id=lambda x: x and x.startswith('articleContent'))[0]
        return str(text_data)


def get_card_links(site, source):
    '''Given the HTML of a webpage from TheTie or The Block, extract
    the hyperlinks to the cards(which lead to articles containing crypto related text).
    '''
    soup = BeautifulSoup(site, 'lxml')

    card_links = []
    if source == 'thetie':
        cards = soup.findAll(
            'article', id=lambda x: x and x.startswith('post-'))
        card_links = [card.header.h3.a['href'] for card in cards]
    elif source == 'block':
        cards = soup.findAll(
            'div', class_=lambda x: x and x.startswith('cardContainer'))
        card_links = ['https://www.theblock.co'+card.a['href']
                      for card in cards]
    return card_links


def get_data(site, data_type, source):
    '''Returns the type of data requested as per source
    Source can be Messari IO, The Block or TheTie.

    Type of data returned can be news articles, scraped text from HTML
    or links to various cards which have the news articles/crypto-related text.
    '''
    hdrs = {'User-Agent': 'Mozilla/39.0'} if source == 'thetie' else {}
    response = requests.get(site, headers=hdrs)

    if data_type == 'text':
        return clean_scraped_text(text_from_html(response, source))
    elif data_type == 'links':
        return get_card_links(response.text, source)
    elif data_type == 'article':
        return clean_article(text_from_html(response, source))
