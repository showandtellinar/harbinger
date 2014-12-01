from flask import Flask
app = Flask(__name__)

from datetime import timedelta

app.config.update(
    CELERY_BROKER_URL='amqp://',
    CELERY_RESULT_BACKEND='amqp',
    CELERY_TIMEZONE = 'America/New_York',
    CELERYBEAT_SCHEDULE = {
        "review-sources": {
            'tasks': 'tasks.load_stories',
            'schedule': timedelta(minutes=5),
        },
    },
)

from celery import Celery

def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery

celery = make_celery(app)

import requests
def get_image(query):
    # base_url="http://api.bing.net/json.aspx?AppId= YOUR_APPID &Version=2.2&Market=en-US&Query=testign&Sources=web+spell&Web.Count=1&JsonType=raw"
    print "Fetching Image: " + query

    base_url = "https://ajax.googleapis.com/ajax/services/search/images"
    res = requests.get(base_url, params={'v': '1.0', 'q': query})

    try:
        data = res.json()["responseData"]["results"]

        if len(data) == 0:
            return None

        return (data[0]["tbUrl"], data[0]["originalContextUrl"], data[0]["title"])
    except:
        return None

@celery.task
def load_stories(file):
    cached_sims = [(s, get_image(s[0][1])) for s in ncluster.main.cluster("ncluster/data/corpus_full.json") if len(s) > 1]
    return cached_sims

from flask import render_template

import ncluster.main

from werkzeug.contrib.cache import SimpleCache
cache = SimpleCache()

from datetime import datetime

def get_or_load_stories():
    cached_sims = cache.get("current-stories")
    if cached_sims == None:
        cached_sims = [(s, get_image(s[0][1])) for s in ncluster.main.cluster("ncluster/data/corpus_full.json") if len(s) > 1]
        cache.set("current-stories", cached_sims, 60 * 10)
    return cached_sims

@app.route('/')
def home():
    cached_sims = get_or_load_stories()
    
    return render_template('main.html', today=datetime.now(), temp="65", sims=sorted(cached_sims, key=lambda s: len(s[0]), reverse=True))

@app.route('/stats')
def stats():
    cached_sims = get_or_load_stories()

    data = {}
    for doc in cached_sims:         
        for s in doc[0]:
            counts = data.setdefault(s[0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            counts[int(round(s[5][0], 1) * 10) - 1] += 1
            data[s[0]] = counts

    return render_template('stats.html', chartData={"datasets":[{"label": s, "data": key} for s, key in data.iteritems()], "labels":["Subjective", "", "", "", "", "", "", "", "", "Objective"]})

if __name__ == "__main__":
    print "Loading Stories..."
    get_or_load_stories()
    print "Finished Loading Stories..."

    app.run(host='0.0.0.0', debug=True)
