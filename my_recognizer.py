import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    all_word_Xlengths = test_set.get_all_Xlengths()

    try:
        for idx in test_set.get_all_sequences():

            X, lengths = all_word_Xlengths[idx]
            word_proabilities = {}
            best_score = float("-inf")
            best_guess = None

            for word, model in models.items():

                try:
                    logL = model.score(X, lengths)
                except:
                    logL = float("-inf")

                word_proabilities[word] = logL

                if logL > best_score:
                    best_score = logL
                    best_guess = word

            probabilities.append(word_proabilities)
            guesses.append(best_guess)

    except Exception as e:
        print(str(e))
        pass
    print(len(probabilities), len(guesses))
    return (probabilities, guesses)
