import preprocessing.preprocessing as preprocessing
import classifiers.multinomial_NB as multinomialNB
import classifiers.bernoulli_NB as bernoulliNB
import classifiers.SVC as SVC
import classifiers.logistic_regression as logistic_regression


def print_divide_line():
    length_of_divide_line = 10
    print("-" * length_of_divide_line)


def apply_all_classifiers(which_comments):
    # 1.
    multinomialNB.multinomial_nb(which_comments)

    # 2.
    bernoulliNB.bernoulli_nb(which_comments)

    # 3.
    SVC.support_vector_classifier(which_comments)
    
    # 4.
    logistic_regression.logistic_regression(which_comments)


def main():

    for i in range(2):
        preprocessing.to_next_phase()
        if i == 0:
            # First phase
            print("===> First phase")
        else:
            # Second phase
            print(sep="\n\n")
            print("===> Second phase")

        # 1.
        preprocessing.without_preprocessing()
        apply_all_classifiers("comments_no_preprocessing.xlsx")
        print_divide_line()

        # 2.
        preprocessing.lower_casing()
        apply_all_classifiers("comments_lower_casing.xlsx")
        print_divide_line()

        # 3.
        preprocessing.tf()
        apply_all_classifiers("comments_tf.xlsx")
        print_divide_line()

        # 4.
        preprocessing.idf()
        apply_all_classifiers("comments_idf.xlsx")
        print_divide_line()

        # 5.
        preprocessing.tf_idf()
        apply_all_classifiers("comments_tf_idf.xlsx")
        print_divide_line()

        # 6.
        preprocessing.stemmering()
        apply_all_classifiers("comments_porter_stemmer.xlsx")
        print_divide_line()

        # 7.
        preprocessing.frequency_word_filtering()
        apply_all_classifiers("comments_frequency_word_filtering.xlsx")
        print_divide_line()

        # 8.
        preprocessing.bigram()
        apply_all_classifiers("comments_bigram.xlsx")
        print_divide_line()

        # 9.
        preprocessing.trigram()
        apply_all_classifiers("comments_trigram.xlsx")
        print_divide_line()


if __name__ == "__main__":
    main()
