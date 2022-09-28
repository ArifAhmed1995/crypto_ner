# Crypto NER
## Extract crypto related keyphrases/words from telegram chat data 

In addition to given data(i.e. `term_def` and `term_abs`) the vocabulary of crypto keyphrases/words is augmented by scraping articles from [Messari IO](https://messari.io/news), [The Block](https://www.theblock.co/category/defi) and [TheTie Research](https://research.thetie.io/category/research/). This is done by using a HuggingFace transformer, a [Keyphrase extraction](https://huggingface.co/spaces/ml6team/keyphrase-extraction) pipeline to extract and save on file the relevant keyphrase/words from the articles after pre-processing the text.

The sentences from the articles are also passed onto a `Word2Vec` model to generate embeddings. Since, `SpaCy` is a popular NLP framework with a simple API, this model is translated to `SpaCy` version and saved on disk.

Possibly relevant noun-phrases are extracted from the chat data. This along with the vocabulary generated earlier can be used to get similarity scores between the noun-phrases and vocabulary words. This is where the `SpaCy` model comes into play by providing the embeddings in order to compute the semantic similarity value between the phrases.

A combined value of lexical and semantic score is used to determine if the keyphrase/word does belong to the crypto domain or not. The filtered ones are returned as output for that message.

### Models
- `./models/crypto.model` - `Word2Vec` model trained on sentences from scraped articles.
- `./models/crypto_model_word2vec.txt.gz` - Zipped text version of the above `Word2Vec` model solely for passing to `SpaCy` 
- `./models/spacy.crypto.word2vec.model` - `SpaCy` version of the above models.


### Datasets
- `./datasets/crypto_sentences.pkl` - Sentences scraped from relevant web articles
- `./datasets/data.csv` - Dataset of telegram messages
- `./datasets/keyphrases.pkl` - Relevant keyphrases that were discovered by the `KeyPhraseExtractor`
- `./datasets/term_abb.csv` - Dataset of crypto terms and their abbreviations
- `./datasets/term_def.csv` - Dataset of crypto terms and their definitions

### Usage

In `crypto_ner.py` end-user methods `get_keyphrase_matches_single` for passing a single message string and `get_keyphrase_matches` for a list of messages. The output is a tuple: `(<original_message>, <list_of_relevant_crypto_keyphrases/words>)`.

#### The initial model loads do take time to execute, so if you have a bunch of messages to extract from please use the `get_keyphrase_matches` method.

### Examples

```
>>> from crypto_ner import get_keyphrase_matches_single, get_keyphrase_matches
>>> get_keyphrase_matches_single('simple public good stablecoin with ETH as backing fee accumulation as collateral')
('simple public good stablecoin with ETH as backing fee accumulation as collateral', ['ETH', 'collateral', 'stablecoin'])
>>> st1 = "This is super exciting. Using deep reinforcement learning to analyze Blockchain security and find even better selfish mining techniques"
>>> st2 = "By the way, which do you think is the on-chain analytics with the best user experience? Preferably the ones that are self-served and that everyone on the team can use"
>>> st3 = "We're launching incentivized testnet on polygon today at tokensoft"
>>> sts = [st1, st2, st3]
>>> kms = get_keyphrase_matches(sts)
>>> for k in kms:
...     print(k)
... 
('This is super exciting. Using deep reinforcement learning to analyze Blockchain security and find even better selfish mining techniques', ['deep reinforcement learning', 'Blockchain security'])
('By the way, which do you think is the on-chain analytics with the best user experience? Preferably the ones that are self-served and that everyone on the team can use', ['on-chain analytics', 'user experience', 'way', 'ones', 'team'])
("We're launching incentivized testnet on polygon today at tokensoft", ['testnet', 'polygon', 'today'])
>>> 

```

### Improvements
- `best_phrases` method in `phrase_tools.py` is the major bottleneck since large language models have to be loaded to find noun-phrases for a sentence.
- The fine tuning for the `KeyPhraseExtractor` needs to be done via a crypto specific dataset. The current model used is fine tuned on scientific literature. This has proved to be better than the one without fine-tuning and other models which used other types of literature.