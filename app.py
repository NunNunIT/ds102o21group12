from flask import Flask, request, render_template_string, jsonify, url_for, current_app
import requests
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
path = os.path.abspath(".")
# sys.path.append('.')
import tensorflow as tf

# Init
tf.compat.v1.enable_eager_execution()
session = tf.compat.v1.Session()

#Thay đổi cái này
# path = "."

#######################################################################################################
# MODEL SVM
import pandas as pd
import re
import numpy as np
# import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, LSTM, Embedding, Bidirectional, GRU
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def preprocessing(data):
    input_string = data

    df = pd.read_csv(f'{path}/Model_SVM/abbreviation_dictionary_vn.csv')
    abbreviation_dict = df.set_index("abbreviation")["meaning"].to_dict()
    abbreviation_dict = {f" {k} ": f" {v} " for k, v in abbreviation_dict.items()}

    # Define the base Vietnamese alphabet without tone marks
    vietnamese_alphabet = "aăâbcdđeêghiklmnoôơpqrstuưvwxy"
    vietnamese_letter_with_tone = "áàạãảắằẵẳặấầẩẫậéèẻẽẹềếểễệòóỏõọồốổỗộờớởỡợúùũủụứừửữựíìĩỉịýỳỹỷỵ"

    # Create uppercase Vietnamese letters with tone marks
    uppercase_vietnamese_letters_with_tone = [char.upper() for char in vietnamese_letter_with_tone]
    uppercase_vietnamese_alphabet = vietnamese_alphabet.upper()

    # Combine the lists into strings
    lowercase_string = vietnamese_alphabet + "".join(vietnamese_letter_with_tone)
    uppercase_string = uppercase_vietnamese_alphabet + "".join(uppercase_vietnamese_letters_with_tone)
    allcase_string = lowercase_string + uppercase_string

    punctuation = "!\"#$%&'()*+,./:;<=>?@[\]^_`{|}~"

    def unicode_replace(text):
        replacements = {
            "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
            "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
            "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
            "Ủy": "Uỷ", "\n": ".", "\t": "."  # Add more replacements as needed
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    input_string = unicode_replace(input_string)

    def remove_emojis_url(text):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F700-\U0001F77F"  # alchemical symbols
                                u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                u"\U00002702-\U000027B0"  # Dingbats
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = url_pattern.sub('', text)

        return text

    input_string = remove_emojis_url(input_string)

    def sticky_preprocess(text):
        result = []
        for letter_id in range(len(text) - 2):
            prev, letter, after = text[letter_id], text[letter_id + 1], text[letter_id + 2]

            if letter in punctuation:
                if prev in allcase_string:
                    result.append(letter_id + 1)
                if after in allcase_string:
                    result.append(letter_id + 2)

        for index in reversed(result):
            text = text[:index] + " " + text[index:]

        return text

    input_string = sticky_preprocess(input_string)

    def abbreviationReplace(text):
        for old, new in abbreviation_dict.items():
            text = text.lower().replace(old, new)
        return text

    input_string = abbreviationReplace(input_string)

    def remove_punctuation_and_numbers(text):
        # Define the regular expression pattern to match punctuation
        pattern = r'[!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^`{|}~\“\”\₫]'

        #lowcase
        text = text.lower()

        # Remove punctuation from the text
        text = re.sub(pattern, '', text)

        return re.sub(r'\s+', ' ', text).strip()

    input_string = remove_punctuation_and_numbers(input_string)

    return input_string


def getLable(data):
    input_string = data
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # def stopwordRemove(text):
    #     for old, new in stopwords_dict.items():
    #         text = text.replace(old, new)
    #     return text

    # input_string = stopwordRemove(input_string)

    def tokenize_string(input_string):

        with open(f'{path}/Model_SVM/tokenizer.json', 'r') as f:
            tokenizer_data = json.load(f)

        tokenizer_config = tokenizer_data['config']
        word_index = tokenizer_data['word_index']

        tok = tokenizer_from_json(tokenizer_config)
        tok.word_index = word_index

        # Tokenize the input string
        tokenized_string = tok.texts_to_sequences([input_string])[0]

        return tok, tokenized_string

    tokenizer, tokenized_string = tokenize_string(input_string)

    def pad_string(max_len, tokenized_string):
        padded_string = pad_sequences([tokenized_string], padding='post', maxlen=max_len)

        return padded_string[0]

    max_len = 64

    padded_string = pad_string(max_len, tokenized_string)

    padded_input = np.reshape(padded_string, (1, max_len))

    csv_file = f'{path}/Model_SVM/embedding_matrix.csv'
    embedding_matrix = pd.read_csv(csv_file, header=None).values

    embedding_dim = 1024

    vocab_aspect_size = len(tokenizer.word_index) + 1

    def Embedding_layer(vocab_aspect_size, embedding_dim, max_len, embedding_matrix, padded_input):
        # Define the input layer
        input = Input(shape=(max_len,))

        # Define the embedding layer with pre-trained weights
        embedding_layer = Embedding(input_dim=vocab_aspect_size,
                                    output_dim=embedding_dim,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    input_length=max_len,
                                    trainable=True)

        # Create a model to get the embeddings
        model = tf.keras.Model(inputs=input, outputs=embedding_layer(input))

        embeded_input = model.predict([padded_input])

        return embeded_input

    embeded_input = Embedding_layer(vocab_aspect_size, embedding_dim, max_len, embedding_matrix, padded_input)

    embeded_input = np.mean(embeded_input, axis=1)

    joblib_file = f'{path}/Model_SVM/svm_model.pkl'
    svm_model_imported = joblib.load(joblib_file)

    single_prediction = svm_model_imported.predict(embeded_input.reshape(1, -1))

    # define the labels
    labels = ["Quality", "Price", "Environment", "Clean", "Personal", "Other"]

    # create a dictionary from the data and labels
    data_dict = dict(zip(labels, single_prediction[0].tolist()))

    # convert the dictionary to JSON
    json_data = json.dumps(data_dict)

    return json_data


#######################################################################################################
# MODEL ENSEMBLE
import py_vncorenlp
from sklearn.pipeline import Pipeline

model_folder = rf'{path}/Model_ENSEMBLE'
vncorenlp_path = os.path.join(model_folder, 'VnCoreNLP')
multi_output_ensemble_path = os.path.join(model_folder, 'multi_output_ensemble.pkl')
tfidf_vectorizer_path = os.path.join(model_folder, 'tfidf_vectorizer.pkl')

class VnCoreNLPSingleton:
    _instance = None

    @staticmethod
    def get_instance(vncorenlp_path):
        if VnCoreNLPSingleton._instance is None:
            VnCoreNLPSingleton._instance = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path, annotators=["wseg"])
        return VnCoreNLPSingleton._instance

def tokenize_text(text, model):
    words = model.annotate_text(text)[0]
    return ' '.join([word["wordForm"] for word in words])

class VnCoreNLPTokenizer:
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(lambda x: tokenize_text(x, self.model))

def create_pipeline(tfidf_vectorizer_path, multi_output_ensemble_path):
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    multi_output_ensemble = joblib.load(multi_output_ensemble_path)

    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('model', multi_output_ensemble)
    ])

    return pipeline

def load_pipeline_and_predict(text):
    vncorenlp = VnCoreNLPSingleton.get_instance(vncorenlp_path)
    pipeline = create_pipeline(tfidf_vectorizer_path, multi_output_ensemble_path)

    tokenized_text = tokenize_text(text, vncorenlp)
    predictions = pipeline.predict([tokenized_text])

    labels = ["Quality", "Price", "Environment", "Clean", "Personal", "Other"]

    data_dict = dict(zip(labels, predictions[0].tolist()))

    return data_dict

#########################################################################################
# HTTPS WEB
app = Flask(__name__)

def run_model_svm(label):
    data = getLable(label)  # Process the JSON data string into a dictionary
    data = json.loads(data)  # Parse JSON data into a dictionary
    response = {
        'Quality': int(data.get('Quality', 0)),
        'Price': int(data.get('Price', 0)),
        'Environment': int(data.get('Environment', 0)),
        'Clean': int(data.get('Clean', 0)),
        'Personal': int(data.get('Personal', 0)),
        'Other': int(data.get('Other', 0))
    }
    return response

def run_model_ensemble(text):
    predictions = load_pipeline_and_predict(text)

    response = {
        'Quality': int(predictions.get('Quality', 0)),
        'Price': int(predictions.get('Price', 0)),
        'Environment': int(predictions.get('Environment', 0)),
        'Clean': int(predictions.get('Clean', 0)),
        'Personal': int(predictions.get('Personal', 0)),
        'Other': int(predictions.get('Other', 0))
    }

    return response

def evaluate_value(value):
    if value == -1:
        return "None"
    elif value == 0:
        return "Negative"
    elif value == 1:
        return "Neutral"
    elif value == 2:
        return "Positive"
    else:
        return "None"


@app.route('/', methods=['GET', 'POST'])
def home():
    svm = None
    ensemble = None
    text_goc = None
    text_after_preproccing = None
    if request.method == 'POST':
        label = request.form.get('data')
        label2 = preprocessing(label)
        
        data = run_model_svm(label2)
        data2 = run_model_ensemble(label2)

        def process_data(data):
            evaluations = {
                "Quality": evaluate_value(data.get('Quality', -1)),
                "Price": evaluate_value(data.get('Price', -1)),
                "Environment": evaluate_value(data.get('Environment', -1)),
                "Clean": evaluate_value(data.get('Clean', -1)),
                "Personal": evaluate_value(data.get('Personal', -1)),
            }
            if all(value == "None" for value in evaluations.values()):
                evaluations["Other"] = "True"
            else:
                evaluations["Other"] = "False"

            return {k: v for k, v in evaluations.items() if v is not None}

        svm = process_data(data)
        ensemble = process_data(data2)
        text_goc = label
        text_after_preproccing = label2
    
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Flask Tailwind</title>
            <style>
                body {
                    margin: 0; 
                    padding: 0;  
                    background: black; 
                    font-family: Arial;              
                }
                .container {
                    min-height: 100vh;
                    height: 100%;
                    display: grid;
                    grid-template-columns: 1fr 1.5fr;
                    gap: 1rem;
                }
                @media (max-width: 768px) {
                    .container {
                        grid-template-columns: 1fr;
                    }
                }
                
                .button {
                    border: none;
                    color: #fff;
                    background-image: linear-gradient(30deg, #0400ff, #4ce3f7);
                    border-radius: 10%;
                    background-size: 100% auto;
                    font-family: inherit;
                    font-size: 1rem;
                    padding: 1.5em;
                    height: 4rem; 
                }

                .button:hover {
                    background-position: right center;
                    background-size: 200% auto;
                    -webkit-animation: pulse 2s infinite;
                    animation: pulse512 1.5s infinite;
                }

                @keyframes pulse512 {
                    0% {
                        box-shadow: 0 0 0 0 #05bada66;
                    }
                    70% {
                        box-shadow: 0 0 0 10px rgb(218 103 68 / 0%);
                    }
                    100% {
                        box-shadow: 0 0 0 0 rgb(218 103 68 / 0%);
                    }
                }
                .loading {
                    position: relative;
                    display: none;
                    font-size: 20px;
                    color: white;
                    text-align: center;
                    margin-top: 5rem;
                }
                
                .spinner {
                position: relative;
                width: 80px;
                /* Adjust the width accordingly */
                /* Keep the height the same */
                }

                .spinner div {
                position: absolute;
                width: 16px;
                height: 16px;
                background-color: #004dff;
                border-radius: 50%;
                animation: spinner-4t3wzl 2s infinite linear;
                }

                .spinner div:nth-child(1) {
                left: 0;
                animation-delay: 0.15s;
                background-color: rgba(0, 77, 255, 0.9);
                }

                .spinner div:nth-child(2) {
                left: 25%;
                animation-delay: 0.3s;
                background-color: rgba(0, 77, 255, 0.8);
                }

                .spinner div:nth-child(3) {
                left: 50%;
                animation-delay: 0.45s;
                background-color: rgba(0, 77, 255, 0.7);
                }

                .spinner div:nth-child(4) {
                left: 75%;
                animation-delay: 0.6s;
                background-color: rgba(0, 77, 255, 0.6);
                }

                .spinner div:nth-child(5) {
                left: 100%;
                animation-delay: 0.75s;
                background-color: rgba(0, 77, 255, 0.5);
                }

                @keyframes spinner-4t3wzl {
                0% {
                    transform: rotate(0deg) translateX(-100%);
                }

                100% {
                    transform: rotate(360deg) translateX(-100%);
                }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div style="padding: 0 1.5rem; width: 90%; height: 100%; background: #18181b; display: flex; flex-direction: column; justify-content: start; align-items: center;">
                    <div>
                        <h1 style="color: white; text-align: center; font-size: 2rem; font-weight: 600; padding: 0.5rem; border-radius: 0.125rem; margin-bottom: 0.5rem;">NHẬP MỘT ĐOẠN TEXT</h1>
                    </div>
                    <form id="myForm" style="width: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;" action="/" method="post">
                        <div style="display: flex; width: 100%; margin: 0 auto;">
                            <div style="width: 100%; max-width: 800px; margin: 0 auto; display: flex; gap: 0.5rem; margin-top: 1.5rem; padding-left: 1rem; padding-right: 1rem;">
                                <textarea id="data" name="data" style="border: 1px solid #cbd5e0; width: 100%; padding: 0.5rem; border-radius: 0.375rem; font-size: 1.25rem; padding-left: 0.5rem; overflow: hidden; resize: none;" placeholder="Nhập một chuỗi" oninput="this.style.height = 'auto'; this.style.height = (this.scrollHeight) + 'px';">{{ result.text if result else '' }}</textarea>
                                <button type="submit" class="button">Run</button>
                            </div>
                        </div>
                    </form>
                    <div style="position: relative; height: 100%; margin: 1rem; padding: 0 1.5rem; width: 100%; background: #27272a; color:white; display: flex; flex-direction:column; justify-content: flex-start; gap: 0.5rem;">
                    <div class="loading">       
                        <div class="spinner">
                        <div></div>
                        <div></div>
                        <div></div>
                        <div></div>
                        <div></div>
                        </div>
                    </div>
                                  
                        <h3 class="textContainer" >Bình luận đang xét:</h3>
                        {% if text_goc %}
                        <p class="textContainer2" style="font-size: 2.5rem; text-align: left; color: white">{{ text_goc }}</p>                                 
                        {% endif %}         
                        <p class="textContainer2" style="font-size: 2.5rem; text-align: left; color: white"></p>
                    </div>
                </div>
                
                <div style="position: sticky; top: 0; right: 0; padding: 0 1.5rem; margin-bottom: 1rem; display: flex; flex-direction: column; gap: 1rem; justify-content: flex-start; height: fit-content;"> 
                    <h2 style="color: white;">Model SVM:</h2>

                    <div style="height: 30vh; padding: 0.25rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; align-items: center; justify-content: center;">
                    <div class="loading">       
                        <div class="spinner">
                        <div></div>
                        <div></div>
                        <div></div>
                        <div></div>
                        <div></div>
                        </div>
                    </div>         
                        {% if svm %}
                            {% for key, value in svm.items() %}
                                <div class="resultContainer" style="padding: 0.75rem; height: 6rem; position: relative; overflow: hidden; background-color: {{ '#dc2626' if value == 'Negative' else '#84cc16' if value == 'Positive' else '#84cc16' if value == 'True' else '#facc15' if value == 'Neutral' else '#a1a1aa' }}; border-radius: 0.5rem; box-shadow: 0 10px 15px rgba(0,0,0,0.1);">
                                    <svg style="position: absolute; bottom: 0; left: 0; margin-bottom: 2rem; transform: scale(1.5); opacity: 0.1;" viewBox="0 0 375 283" fill="none">
                                        <rect x="159.52" y="175" width="152" height="152" rx="8" transform="rotate(-45 159.52 175)" fill="white" />
                                        <rect y="107.48" width="152" height="152" rx="8" transform="rotate(-45 0 107.48)" fill="white" />
                                    </svg>
                                    <div style="position: relative; height: 100%; display: flex; flex-direction: column; justify-content: space-between; gap: 0.25rem; color: white;">
                                        <span style="display: block; font-weight: 600; color: black; margin-bottom: -0.25rem; font-size: 2rem;">{{ key }}</span>
                                        {% if value %}
                                            <span class="value" style="display: block; font-weight: 600; font-size: 2.5rem; text-align:right;">{{ value }}</span>
                                        {% else %}
                                            <span class="value" style="display: block; font-weight: 600; font-size: 2.5rem; text-align:right;">None</span>
                                        {% endif %}
                                    </div>
                                </div>
                            {% endfor %}
                        {% endif %}
                    </div>

                    <h2 style="color: white;">Model ENSEMBLE:</h2>
                                  
                    <div style="height: 30vh; padding: 0.25rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; align-items: center; justify-content: center;">
                        <div class="loading">       
                            <div class="spinner">
                            <div></div>
                            <div></div>
                            <div></div>
                            <div></div>
                            <div></div>
                            </div>
                        </div>
                        {% if ensemble %}
                            {% for key, value in ensemble.items() %}
                                <div class="resultContainer" style="padding: 0.75rem; height: 6rem; position: relative; overflow: hidden; background-color: {{ '#dc2626' if value == 'Negative' else '#84cc16' if value == 'Positive' else '#84cc16' if value == 'True' else '#facc15' if value == 'Neutral' else '#a1a1aa' }}; border-radius: 0.5rem; box-shadow: 0 10px 15px rgba(0,0,0,0.1);">
                                    <svg style="position: absolute; bottom: 0; left: 0; margin-bottom: 2rem; transform: scale(1.5); opacity: 0.1;" viewBox="0 0 375 283" fill="none">
                                        <rect x="159.52" y="175" width="152" height="152" rx="8" transform="rotate(-45 159.52 175)" fill="white" />
                                        <rect y="107.48" width="152" height="152" rx="8" transform="rotate(-45 0 107.48)" fill="white" />
                                    </svg>
                                    <div style="position: relative; height: 100%; display: flex; flex-direction: column; justify-content: space-between; gap: 0.25rem; color: white;">
                                        <span style="display: block; font-weight: 600; color: black; margin-bottom: -0.25rem; font-size: 2rem;">{{ key }}</span>
                                        {% if value %}
                                            <span class="value" style="display: block; font-weight: 600; font-size: 2.5rem; text-align:right;">{{ value }}</span>
                                        {% else %}
                                            <span class="value" style="display: block; font-weight: 600; font-size: 2.5rem; text-align:right;">None</span>
                                        {% endif %}
                                    </div>
                                </div>
                            {% endfor %}
                        {% endif %}
                    </div>
                   
                </div>
            </div>
        <script>
            document.getElementById('myForm').addEventListener('submit', function() {
                var formData = new FormData(event.target); // Get form data
                var textInput = formData.get('data');
                document.querySelector('.textContainer2').style.display = 'block';
                document.querySelector('.textContainer2').textContent = textInput;               
                                  
                document.querySelectorAll('.loading').forEach(function(element) {
                    element.style.display = 'block'; // Show all elements with class 'loading'
                });
                document.querySelectorAll('.spinner').forEach(function(element) {
                    element.style.display = 'block'; // Show all elements with class 'spinner'
                });
                document.querySelector('.textContainer').style.display = 'none'; // Hide the element with class 'textContainer'
                document.querySelectorAll('.resultContainer').forEach(function(element) {
                    element.style.display = 'none'; // Hide all elements with class 'resultContainer'
                });
                
            });
        </script>
        </body>
        </html>
    ''', text_goc=text_goc, text_after_preproccing=text_after_preproccing, svm=svm, ensemble=ensemble)


if __name__ == '__main__':
    app.run(debug=True)

