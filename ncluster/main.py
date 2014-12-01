import feedparser, requests, html2text, nltk
import sys, codecs, os

from textblob import TextBlob

import traceback

# Summarize
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers.czech import stem_word
from sumy.utils import get_stop_words

currentFeeds = [
    ("BBC", "http://feeds.bbci.co.uk/news/rss.xml"),
    ("CNN", "http://rss.cnn.com/rss/cnn_topstories.rss"),
    ("USATODAY", "http://rssfeeds.usatoday.com/usatoday-newstopstories&x=1"),
    ("ABC", "http://feeds.abcnews.com/abcnews/topstories"),
    ("Al Jazeera", "http://www.aljazeera.com/Services/Rss/?PostingId=2007731105943979989"),
    ("FOX", "http://feeds.foxnews.com/foxnews/latest?format=xml"),
    # ("Washington Post World", "http://feeds.washingtonpost.com/rss/world"),
    ("Washington Post National", "http://feeds.washingtonpost.com/rss/national"),
    ("New York Times", "http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
    ("The Economist", "http://www.economist.com/sections/united-states/rss.xml"),
    ("NPR", "http://www.npr.org/rss/rss.php?id=1001"),
]

def set_sources():
    newFeeds = []
    while True:
        source = raw_input("Source as Title,Source URL (or 'quit' to stop): ")
        if source == "quit":
            print "Set Sources to", newFeeds
            return newFeeds

        data = source.split(",")
        newFeeds.append((data[0], data[1]))

def download_sources(summarize=True, sources=currentFeeds):
    raw_documents = []
    complete_urls = []

    # Download News Stories
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.bypass_tables = True

    count_error = 0
    document_count = 0

    feed_count = -1

    for url in currentFeeds:
        feed_count += 1
        current_feed_document = 0

        currentStories = []
        feed = feedparser.parse(url[1])
        for story in feed.entries:
            current_feed_document += 1

            if story.title.startswith(u'VIDEO:') or story.title.startswith(u'AUDIO'):
                continue
            if story.link in complete_urls:
                continue

            try:
                res = requests.get(story.link)

                html = res.text
                title = story.title.encode('utf-8')
                
                completion = ((feed_count + (current_feed_document / float(len(feed.entries)))) / (float(len(currentFeeds))))* 100
                
                print "[" + ("%.2f" % completion) + "%] \t " + feed.feed.title.encode('utf-8') + " - " + title

                raw_text = converter.handle(html)
                if summarize:
                    parser = HtmlParser.from_string(html, None, Tokenizer("english"))
                
                    summarizer = LsaSummarizer(stem_word)
                    summarizer.stop_words = get_stop_words("english")

                    sum_text = [sentence for sentence in summarizer(parser.document, 20)]
                    raw_text = (" ".join([str(sentence) for sentence in sum_text])).decode('utf-8')
                    # print raw_text

                stats = TextBlob(raw_text)
                currentStories.append((title, raw_text, story.link, stats.sentiment, story.published_parsed))
                complete_urls.append(story.link)

                document_count += 1

            except KeyboardInterrupt:
                print "Quitting from Keyboard Interrupt."
                sys.exit(0)
            except:
                count_error += 1
                print "\t Error occurred while processing that story:", sys.exc_info()[0]
                traceback.print_exc()

        raw_documents.append((url[0], currentStories))

    print "Received", document_count, "documents with", count_error, "errors"
    return raw_documents

def sentiment_analysis(source):
    return TextBlob(source).sentiment

# Process Corpora
def prepare_corpus(raw_documents):
    # remove punctuation
    print "Removing Punctuation"
    import string
    exclude = set(string.punctuation)
    raw_documents = [''.join(ch for ch in s if ch not in exclude) for s in raw_documents]

    # remove common words
    print "Calculating Stoplist"
    stoplist = set([x.rstrip() for x in codecs.open("stop_list.txt", encoding='utf-8') if not x.startswith("#")])
    stoplist = stoplist.union(set(nltk.corpus.stopwords.words("english")))
    # print stoplist

    print "Removing Stoplist and Stemming"

    from nltk.stem.lancaster import LancasterStemmer
    st = LancasterStemmer()

    texts = [[st.stem(word) for word in document.lower().split() if word not in stoplist]
             for document in raw_documents]

    # remove words that appear only once
    print "Removing Single Variables"
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once]
             for text in texts]

    return texts

from gensim import corpora, models, similarities, matutils

NUM_TOPICS = 300

def run_gensim(texts, pr, lsi):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = tfidf[corpus]

    model = None
    if not lsi == None:
        model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)
    else:
        model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=NUM_TOPICS)

    if pr:
        model.print_topics(num_topics=-1)

    return (model, dictionary, corpus_tfidf)

def get_sim(lda, corpus, dictionary, data, query):
    vec_bow = dictionary.doc2bow(query.lower().split())
    vec_lsi = lda[vec_bow]

    index = similarities.MatrixSimilarity(lda[corpus])

    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    sims = [(data[x[0]][0], data[x[0]][1], x[1], data[x[0]][2], data[x[0]][3], data[x[0]][4]) for x in sims if x[1] > 0.2]

    # for docs in sims:
    #    print docs
    return sims

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls

def plot_sentiment(credentials, filename, sentiment, prop):
    access = lambda sen: sen.polarity
    if prop == "subjectivity":
        access = lambda sen: sen.subjectivity

    py.sign_in(credentials[0], credentials[1])

    data = Data(
        [Histogram(
            x=[access(story) for story in x[1]],
            name=x[0],
            histnorm="percent",
            autobinx=False,
            xbins=XBins(
                start=-1.1,
                end=1.1,
                size=0.05,
            )
        )
    for x in sentiment])

    layout = Layout(
        title=prop + " of news sources.",
        barmode='stack'
    )
    fig = Figure(data=data, layout=layout)

    plot_url = py.plot(fig, filename=filename)
    print "Your plot.ly is done at", plot_url

from itertools import chain

def try_cluster_1(l, lda_corpus, d, documents):
    scores = list(chain(*[[score for topic,score in topic] \
                          for topic in [doc for doc in lda_corpus]]))
    threshold = sum(scores)/len(scores)
    threshold = 0.2
    print "Threshold: ", threshold
    print 

    for topic in range(len(lda_corpus)):
        cluster = [j for i,j in zip(lda_corpus,documents) if len(i) > topic and i[topic][1] > threshold]
        print cluster

def try_cluster_2(l, lda_corpus, d, documents):
    from sklearn.cluster import KMeans
    kmeans = KMeans(20).fit(matutils.corpus2dense(lda_corpus, NUM_TOPICS))
    print kmeans.labels_

def try_cluster_3(lda, corpus, dictionary, documents):
    done = []
    clusters = []

    for doc in documents:
        if doc in done:
            continue

        sims = []
        for s in get_sim(lda, corpus, dictionary, documents, doc[2]):
            if not (s[0], s[1]) in done:
                done.append((s[0], s[1]))
                sims.append(s)
        
        if len(sims) > 0:
            clusters.append(sims)
            
    return clusters

def cluster(filename=None, use_lsi=True):
    tempData = load_corpus(filename)
    data = tempData[0]
    sentiment = tempData[1]
    texts = tempData[2]
    credentials = tempData[3]

    lda, dictionary, processed_corpus = run_gensim(texts, True, use_lsi)
    sims = try_cluster_3(lda, processed_corpus, dictionary, [(source[0], story[0], story[1], story[2], story[3]) for source in data for story in source[1]])
    return sims

import json

def save_corpus(corpus, filename):
    with open(filename, 'w') as outfile:
        json.dump(corpus, outfile)

def load_corpus(filename):
    with open(filename, 'r') as outfile:
        return json.load(outfile)

if __name__ == "__main__":
    print "Hello."

    # Global Variables
    data = None
    sentiment = None
    texts = None
    credentials = None

    lda = None
    processed_corpus = None
    dictionary = None

    uname = os.getenv("PLOTLY_USERNAME", None)
    api = os.getenv("PLOTLY_API", None)
    if uname != None and api != None:
        credentials = [uname, api]

    # REPL Loop
    while True:
        command = raw_input("> ")

        if command == "quit":
            break
        elif command == "download":
            data = download_sources()
        elif command == "save":
            if data == None:
                print "Need to run 'download' or 'load' first."
                continue

            where = raw_input("Filename: ")
            try:
                save_corpus([
                    data,
                    sentiment,
                    texts,
                    credentials,
                    lda,
                    processed_corpus,
                    dictionary,
                ], where)
            except:
                print "Unable to save data.", sys.exc_info()[0]
        elif command == "load":
            where = raw_input("Filename: ")

            try:
                tempData = load_corpus(where)
                data = tempData[0]
                sentiment = tempData[1]
                texts = tempData[2]
                credentials = tempData[3]
                lda = tempData[4]
                processed_corpus = tempData[5]
                dictionary = tempData[6]
            except:
                print "Unable to load data.", sys.exc_info()[0]
        elif command == "sentiment":
            if data == None:
                print "Need to run 'download' or 'load' first."
                continue

            sentiment = [(source[0], [sentiment_analysis(story[1]) for story in source[1]]) for source in data]
        elif command == "plot":
            if sentiment == None:
                print "Neet to run 'sentiment' first."
                continue

            if credentials == None:
                username = raw_input("Plotly Username: ")
                api_key = raw_input("Plotly API Key: ")
                credentials = [username, api_key]

            filename = raw_input("Plotly Filename: ")
            prop = raw_input("Property of Data (polarity or subjectivity): ")

            plot_sentiment(credentials, filename, sentiment, prop)
        elif command == "prepare":
            if data == None:
                print "Need to run 'download' or 'load' first."
                continue

            texts = prepare_corpus([story[1] for source in data for story in source[1]])
        elif command == "gensim":
            if texts == None:
                print "Need to run 'prepare' first."
                continue

            use_lsi = raw_input("Want to use LSI (y or nothing)? ")
            if use_lsi == "":
                use_lsi = None
            else:
                use_lsi = True

            lda, dictionary, processed_corpus = run_gensim(texts, True, use_lsi)
        elif command == "getsim":
            if lda == None:
                print "Need to run 'gensim' first."
                continue

            query = raw_input("Query: ")

            get_sim(lda, processed_corpus, dictionary, [(source[0], story[0], story[1]) for source in data for story in source[1]], query)
        elif command == "cluster":
            if lda == None:
                print "Need to run 'gensim' first."
                continue

            sims = try_cluster_3(lda, processed_corpus, dictionary, [(source[0], story[0], story[1]) for source in data for story in source[1]])
            for s in sims:
                print s
        elif command == "setsources":
            currentFeeds = set_sources()
        elif command == "help":
            print "quit"
            print "download"
            print "save"
            print "load"
            print "sentiment"
            print "prepare"
            print "gensim"

    print "Complete"
