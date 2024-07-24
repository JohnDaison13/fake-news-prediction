from flask import Flask,render_template,request, redirect, url_for,session
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

model = pickle.load(open('fake_news_model.pkl','rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

app = Flask(__name__)
app.secret_key = 'my_super_secret_key'

port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('^[a-zA-Z]','',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

def preprocess_and_predict(text):
    stemmed_text = stemming(text)
    text_vectorized = vectorizer.transform([stemmed_text]).toarray()
    prediction = model.predict(text_vectorized)
    return 'Fake' if prediction[0] == 1 else 'Real'

@app.route('/')
def home():
    prediction = session.pop('prediction', None)
    news_title = session.get('news_title', '') 
    return render_template('home.html', data=prediction, news_title=news_title)

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['news']
    prediction = preprocess_and_predict(title)
    session['prediction'] = prediction
    session['news_title'] = title
    return redirect(url_for('home'))

@app.route('/clear',methods=['POST'])
def clear():
    session.pop('news_title', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

