
import os
import warnings
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import bentoml
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download('omw-1.4')

def load_train_test_data(data_url:str):
    import os
    data = pd.read_csv(data_url)

    X = data.Comment
    y = data['Sous catégorie 1']

    return train_test_split(X, y, test_size=0.2)


def create_model(**kwargs):
    from sklearn.base import BaseEstimator
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline

    class TextPreprocessor(BaseEstimator):
        """TextPreprocessor preprocesses text by applying these rules:

        - Remove special chars
        - Remove punctuation
        - Convert to lowercase
        - Replace numbers
        - Tokenize text
        - Remove stopwords
        - Lemmatize words

        It implements the BaseEstimator interface and can be used in sklearn pipelines.
        """

        def remove_special_chars(self, text):
            import re
            import html

            re1 = re.compile(r'  +')
            x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
                'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
                '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
                ' @-@ ', '-').replace('\\', ' \\ ')
            return re1.sub(' ', html.unescape(x1))

        def remove_punctuation(self, text):
            """Remove punctuation from list of tokenized words"""
            import string

            translator = str.maketrans('', '', string.punctuation)
            return text.translate(translator)

        def to_lowercase(self, text):
            return text.lower()

        def replace_numbers(self, text):
            """Replace all interger occurrences in list of tokenized words with textual representation"""
            import re

            return re.sub(r'\d+', '', text)

        def text2words(self, text):
            from nltk.tokenize import word_tokenize

            return word_tokenize(text)

        def remove_stopwords(self, words):
            """
            :param words:
            :type words:
            :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
            or
            from spacy.lang.en.stop_words import STOP_WORDS
            :type stop_words:
            :return:
            :rtype:
            """
            from nltk.corpus import stopwords
            stop_words = stopwords.words('french')

            return [word for word in words if word not in stop_words]

        def lemmatize_words(self, words):
            """Lemmatize words in text"""
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(word) for word in words]

        def lemmatize_verbs(self, words):
            """Lemmatize verbs in text"""
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

        def clean_text(self, text):
            text = self.remove_special_chars(text)
            text = self.remove_punctuation(text)
            text = self.to_lowercase(text)
            text = self.replace_numbers(text)
            words = self.text2words(text)
            words = self.remove_stopwords(words)
            words = self.lemmatize_words(words)
            words = self.lemmatize_verbs(words)

            return ''.join(words)

        def fit(self, x, y=None):
            return self

        def transform(self, x):
            return map(lambda x: self.clean_text(x), x)


    return Pipeline([
        ('preprocessing', TextPreprocessor()),
        ('tfid', TfidfVectorizer(ngram_range=(1, 3),max_features=100)),
        ('RandomForest', RandomForestClassifier())
    ])



def run_workflow(tracking_server_url: str, mlflow_experiment_name: str, mlflow_run_name: str, data_url: str):
    # Step1: Prepare data
    train_X, test_X, train_y, test_y = load_train_test_data(data_url)
   
    if tracking_server_url:
        mlflow.set_tracking_uri(tracking_server_url)
    if mlflow_experiment_name:
        mlflow.set_experiment(mlflow_experiment_name)


    # Step2: Train the model within the mlflow context
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        # create a random forest classifier
        model = create_model()

    #
        # train the model with training_data
        #rf_clf.fit(train_X, train_y)
        model.fit(train_X, train_y)
        model_accuracy=model.score(train_X, train_y)
        # predict testing data
        #predicts_val = rf_clf.predict(test_X)
        
        # step3: Track the model
        print("accuracy: %f" % model_accuracy)
        # log the hyper-parameter to mlflow tracking server
        mlflow.log_param("data_url", data_url)
        #mlflow.log_param("n_estimator", n_estimator)
      # mlflow.log_param("max_depth", max_depth)
      # mlflow.log_param("min_samples_split", min_samples_split)
        # log shap feature explanation extension. This will generate a graph of feature importance of the model
        # mlflow.shap.log_explanation(rf_clf.predict, test_X.sample(70))
        # log the metric
        mlflow.log_metric("model_accuracy", model_accuracy)
        # log the model
        extra_pip_requirements = ["nltk", "numpy"]
        mlflow.sklearn.log_model(
            model, "model", extra_pip_requirements=extra_pip_requirements)

        #mlflow.sklearn.log_model(rf_clf, "model")
        model_uri = mlflow.get_artifact_uri("model")

        print('depedence de model à vérifier                 :',mlflow.pyfunc.get_model_dependencies(model_uri))
        print('model uri :',model_uri)
        bentoml.mlflow.import_model(
            "comment-classifier",
            model_uri,
            signatures={"predict": {"batchable": True}})
     #   mlflow.sklearn.log_model(rf_clf, "model")
def main():
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # default configuration
    default_mlflow_server_url = "https://localhost:5000/"
    default_data_url = "https://minio.lab.sspcloud.fr/mabdelli/mlflow/exercice1.csv"
    default_experiment_name = "Projet_test"
    default_run_name = "default"

   # default_n_estimator = 10
   # default_max_depth = 5
   # default_samples_split = 2

    # Get experiment setting from cli
    remote_server_uri = str(sys.argv[1]) if len(sys.argv) > 1 else default_mlflow_server_url
    experiment_name = str(sys.argv[2]) if len(sys.argv) > 2 else default_experiment_name
    run_name = str(sys.argv[3]) if len(sys.argv) > 3 else default_run_name

    # Get data path
    data_url = str(sys.argv[4]) if len(
        sys.argv) > 4 else default_data_url

    # Get hyper parameters from cli arguments
    #n_estimator = int(sys.argv[5]) if len(sys.argv) > 5 else default_n_estimator
   # max_depth = int(sys.argv[6]) if len(sys.argv) > 6 else default_max_depth
 #   min_samples_split = int(sys.argv[7]) if len(sys.argv) > 7 else default_samples_split

    # split data into training_data and test_data

    run_workflow(remote_server_uri, experiment_name, run_name, data_url)


if __name__ == "__main__":
    main()
