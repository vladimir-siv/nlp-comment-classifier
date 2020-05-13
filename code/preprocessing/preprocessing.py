# Util classes
import pandas as pd
import re
import numpy as np

from numpy import mean
from math import isnan

# Stemming classes
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Preprocessing classes
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


first_phase = False
X = None


def get_data_set():
    return X


def to_next_phase():
    global first_phase
    first_phase = not first_phase


def remove_string_special_characters(s):
    # removes special characters with ' '
    stripped = re.sub('[^a-zA-z\s]', ' ', s)
    stripped = re.sub('_', ' ', stripped)

    # Change any white space and new line to one space
    stripped = stripped.replace('\\n', ' ')
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
        return stripped


def bag_of_words_to_int_data_set(data_frame, gram_range):
    # Calculate matrix of occurrence of every word, bigram or trigram (based on value of gram_range)
    global X
    count_vectorizer = CountVectorizer(ngram_range=gram_range)
    X = count_vectorizer.fit_transform(data_frame['Comment'])


def set_data_set(data_set):
    global X
    X = data_set


def prepare_type_for_first_phase(t):
    if t in {'Functional-Inline', 'Functional-Method', 'Functional-Module'}:
        t = 'Functional'
    return t


def remove_outliers(data_frame):
    # Count unigrams with CountVectorizer
    count_vectorizer = CountVectorizer(ngram_range=(1, 1))
    unigram_counts = count_vectorizer.fit_transform(data_frame['Comment']).toarray()

    # Count number of words and mean number of words per data frame row
    number_of_words = 0
    for row_index, row in enumerate(unigram_counts):
        for column_index, column in enumerate(row):
            if unigram_counts[row_index][column_index] > 0:
                number_of_words += unigram_counts[row_index][column_index]

    mean_num_words = number_of_words / data_frame.shape[0]

    def cut_long_outlier(comment):
        words = word_tokenize(comment)
        return "".join([word + " " for index, word in enumerate(words) if index <= mean_num_words])

    data_frame['Comment'] = data_frame['Comment'].apply(cut_long_outlier)

    return data_frame


def init_preprocessing():
    # Reading data and remove unused
    data_frame = pd.read_csv("raw_data/comments.txt", sep="\t", names=['NaturalLanguageID', 'ProgrammingLanguageName',
                                                                       'RepoID', 'SourceID', 'CommentID', 'Comment',
                                                                       'Type'])
    data_frame = data_frame[['Comment', 'Type']]

    # Remove NA values, special characters, outliers, duplicates.
    data_frame = data_frame.dropna()
    data_frame['Comment'] = data_frame['Comment'].apply(remove_string_special_characters)
    if first_phase:
        data_frame['Type'] = data_frame['Type'].apply(prepare_type_for_first_phase)
    data_frame = remove_outliers(data_frame)
    data_frame = data_frame.drop_duplicates()

    # Transforming types to integer values
    if first_phase:
        labels = ['Functional', 'ToDo', 'Notice', 'General', 'Code', 'IDE']
    else:
        labels = ['Functional-Inline', 'Functional-Method', 'Functional-Module', 'ToDo', 'Notice', 'General', 'Code',
                  'IDE']
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    data_frame['Type'] = label_encoder.transform(data_frame['Type'])

    return data_frame


def without_preprocessing():
    print("==> No preprocessing")

    data_frame = init_preprocessing()
    bag_of_words_to_int_data_set(data_frame, (1, 1))

    data_frame.to_excel(r'preprocessed_data\comments_no_preprocessing.xlsx', index=False)


def lower_casing():
    print("==> Lower casing")

    data_frame = init_preprocessing()
    data_frame['Comment'] = data_frame['Comment'].apply(lambda comment: comment.lower())
    bag_of_words_to_int_data_set(data_frame, (1, 1))

    data_frame.to_excel(r'preprocessed_data\comments_lower_casing.xlsx', index=False)


def tf():
    print("==> Term Frequency")
    data_frame = init_preprocessing()

    # Count unigrams with CountVectorizer
    count_vectorizer = CountVectorizer(ngram_range=(1, 1))
    unigram_counts = count_vectorizer.fit_transform(data_frame['Comment'])

    # Calculate TF value for each comment
    tf_transformer = TfidfTransformer(use_idf=False)
    tf_counts = tf_transformer.fit_transform(unigram_counts)

    # Set new data set
    set_data_set(tf_counts)

    data_frame.to_excel(r'preprocessed_data\comments_tf.xlsx', index=False)


def idf():
    print("==> Inverse Document Frequency")
    data_frame = init_preprocessing()

    # Count unigrams with CountVectorizer
    count_vectorizer = CountVectorizer(ngram_range=(1, 1))
    unigram_counts = count_vectorizer.fit_transform(data_frame['Comment'])

    # Calculate IDF value for each comment, using formula IDF=TF_IDF / TF
    tf_idf_transformer = TfidfTransformer(use_idf=True)
    tf_idf_counts = tf_idf_transformer.fit_transform(unigram_counts).toarray()

    tf_transformer = TfidfTransformer(use_idf=False)
    tf_counts = tf_transformer.fit_transform(unigram_counts).toarray()
    tf_counts[tf_counts == 0] = 1

    idf_counts = np.divide(tf_idf_counts, tf_counts)

    # Set new data set
    set_data_set(idf_counts)

    data_frame.to_excel(r'preprocessed_data\comments_idf.xlsx', index=False)


def tf_idf():
    print("==> Term Frequency–Inverse Document Frequency")
    data_frame = init_preprocessing()

    # Calculate TF value for each comment
    tf_idf_transform = TfidfVectorizer(ngram_range=(1, 1))
    tf_idf_counts = tf_idf_transform.fit_transform(data_frame['Comment'])

    # Set new data set
    set_data_set(tf_idf_counts)

    data_frame.to_excel(r'preprocessed_data\comments_tf_idf.xlsx', index=False)


def stemmering():
    print("==> PorterStemmer")
    data_frame = init_preprocessing()
    ps = PorterStemmer()

    def tokenize_stem_and_remove_stop_words(comment):
        words = word_tokenize(comment)
        return "".join([ps.stem(word) + " " for word in words if word not in stopwords.words()])

    data_frame['Comment'] = data_frame['Comment'].apply(tokenize_stem_and_remove_stop_words)

    bag_of_words_to_int_data_set(data_frame, (1, 1))

    data_frame.to_excel(r'preprocessed_data\comments_porter_stemmer.xlsx', index=False)


def frequency_word_filtering():
    print("==> Frequency word filtering")
    data_frame = init_preprocessing()

    # Count unigrams with CountVectorizer
    tf_transform = TfidfVectorizer(use_idf=False, ngram_range=(1, 1))
    tf_counts = tf_transform.fit_transform(data_frame['Comment']).toarray()
    features = tf_transform.get_feature_names()

    # Mean TF value with correction factor
    mean_value = mean([elem for row in tf_counts for elem in row if (elem > 0)]) * 0.75

    # All values with frequency below mean value reset to zero.
    tf_counts[tf_counts < mean_value] = 0

    # Remove all element below mean value and store them in new data frame
    new_data_frame = pd.DataFrame(columns=['Comment', 'Type'])

    def create_unigrams_from_comments():
        for row_index, row in enumerate(tf_counts):
            for column_index, column in enumerate(row):
                if column >= mean_value:
                    new_data_frame.loc[0 if pd.isnull(new_data_frame.index.max()) else new_data_frame.index.max() + 1] \
                        = [features[column_index], list(data_frame['Type'])[row_index]]

    create_unigrams_from_comments()
    new_data_frame = new_data_frame.drop_duplicates()

    # Set new data set
    set_data_set(tf_counts)

    data_frame.to_excel(r'preprocessed_data\comments_frequency_word_filtering.xlsx', index=False)
    new_data_frame.to_excel(r'preprocessed_data\comments_frequency_word_filtering_view.xlsx', index=False)


def bigram():
    print("==> Bigram preprocessing")
    data_frame = init_preprocessing()

    bigram_transform = CountVectorizer(ngram_range=(2, 2))
    bigram_counts = bigram_transform.fit_transform(data_frame['Comment'])
    bigram_counts_indices = bigram_counts.toarray()
    features = bigram_transform.get_feature_names()

    new_data_frame = pd.DataFrame(columns=['Comment', 'Type'])

    def create_bigrams_from_comments():
        for row_index, row in enumerate(bigram_counts_indices):
            for column_index, column in enumerate(row):
                if column == 1:
                    new_data_frame.loc[0 if pd.isnull(new_data_frame.index.max()) else new_data_frame.index.max() + 1] \
                        = [features[column_index], list(data_frame['Type'])[row_index]]

    create_bigrams_from_comments()
    new_data_frame['Comment'] = new_data_frame['Comment'].drop_duplicates()

    # Set new data set.
    set_data_set(bigram_counts)

    data_frame.to_excel(r'preprocessed_data\comments_bigram.xlsx', index=False)
    new_data_frame.to_excel(r'preprocessed_data\comments_bigrams.xlsx', index=False)


def trigram():
    print("==> Trigram preprocessing")
    data_frame = init_preprocessing()

    trigram_transform = CountVectorizer(ngram_range=(3, 3))
    trigram_counts = trigram_transform.fit_transform(data_frame['Comment'])
    trigram_transform_indices = trigram_counts.toarray()
    features = trigram_transform.get_feature_names()

    new_data_frame = pd.DataFrame(columns=['Comment', 'Type'])

    def create_trigrams_from_comments():
        for row_index, row in enumerate(trigram_transform_indices):
            for column_index, column in enumerate(row):
                if column == 1:
                    new_data_frame.loc[0 if pd.isnull(new_data_frame.index.max()) else new_data_frame.index.max() + 1] \
                        = [features[column_index], list(data_frame['Type'])[row_index]]

    create_trigrams_from_comments()
    new_data_frame = new_data_frame.drop_duplicates()

    # Set new data set.
    set_data_set(trigram_counts)

    data_frame.to_excel(r'preprocessed_data\comments_trigram.xlsx', index=False)
    new_data_frame.to_excel(r'preprocessed_data\comments_trigrams_view.xlsx', index=False)
