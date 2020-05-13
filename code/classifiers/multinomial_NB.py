# Util classes
import pandas as pd

# Classifier class
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB

# Data set
import preprocessing.preprocessing as preprocessing


def multinomial_nb(which_comments):
    print("=> Multinomial Bayes naive classifier")

    data_frame = pd.read_excel("preprocessed_data/{}".format(which_comments))
    x = preprocessing.get_data_set()

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
    index = 1
    for train_index, test_index in sss.split(x, data_frame['Type']):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = data_frame['Type'][train_index], data_frame['Type'][test_index]

        mnb = MultinomialNB()
        mnb.fit(x_train, y_train)
        score = mnb.score(x_test, y_test)

        print("Score {}.: {:.2f}%".format(index, score * 100), end=" ")
        if index == 5:
            print()
        index += 1

    print()
