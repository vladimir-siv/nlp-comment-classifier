# Util classes
import pandas as pd
import numpy as np

# Classifier class
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from nested_cv import NestedCV


# Data set
import preprocessing.preprocessing as preprocessing


def compare_regularisation_functions(data_frame, rf, c=1.0):
    x = preprocessing.get_data_set()

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
    index = 1
    average = 0
    for train_index, test_index in sss.split(x, data_frame['Type']):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = data_frame['Type'][train_index], data_frame['Type'][test_index]

        if rf == 'l1':
            solver = 'saga'
        else:
            solver = 'lbfgs'

        lr = LogisticRegression(penalty=rf, C=c, solver=solver, max_iter=15000)
        lr.fit(x_train, y_train)

        score = lr.score(x_test, y_test)
        average = average + score
        print("Score({}) {}.: {:.2f}%".format(rf.upper(), index, score * 100), end=" ")

        if index == 5:
            print()
        index += 1

    print()
    print("Average: {:.2f}%".format(average / 10 * 100))


def optimize_c_parameter(data_frame):
    models_param = {
        'max_iter': [15000],
        'C': [1]
    }

    nested_cv_search = NestedCV(model=LogisticRegression(), params_grid=models_param,
                                outer_kfolds=5, inner_kfolds=5,
                                cv_options={'sqrt_of_score': True, 'randomized_search_iter': 30})

    x = preprocessing.get_data_set()
    nested_cv_search.fit(x, data_frame['Type'])

    optimized_c_value = np.mean(nested_cv_search.outer_scores)
    print("Optimized C: {:.3f}".format(optimized_c_value))
    compare_regularisation_functions(data_frame, 'l2', optimized_c_value)


def logistic_regression(which_comments):
    print("=> Logistic regression")
    data_frame = pd.read_excel("preprocessed_data/{}".format(which_comments))

    # Testing differences between regularisation functions
    print("> L1/L2 comparing")
    compare_regularisation_functions(data_frame, 'l1')
    compare_regularisation_functions(data_frame, 'l2')

    # Optimizing C parameter
    print("> Results with optimized C parameter")
    optimize_c_parameter(data_frame)
