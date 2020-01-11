import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score

def model_check(true_labels, predictions):
    '''Model performance mesaurement'''
    if type(true_labels) is tuple:
        # Printing header row
        print('%11s%11s%11s%11s' % ('Accuracy', 'Precision', 'Recall', 'F1'))
        print('%s' % '-'*44)
        for i in [0, 1]:
            y_true = true_labels[i]
            y_pred = predictions[i]
            print('%11.5f%11.5f%11.5f%11.5f%10s' % \
                      (accuracy_score(y_true, y_pred), \
                       precision_score(y_true, y_pred), \
                       recall_score(y_true, y_pred), \
                       f1_score(y_true, y_pred),  \
                       ['<-- train', '<-- test'][i]))
    else:
        y_true = true_labels
        y_pred = predictions
        print('%11s%11s%11s%11s' % ('Accuracy', 'Precision', 'Recall', 'F1'))
        print('%s' % '-'*44)
        print('%11.5f%11.5f%11.5f%11.5f' % (accuracy_score(y_true, y_pred), \
                                            precision_score(y_true, y_pred), \
                                            recall_score(y_true, y_pred), \
                                            f1_score(y_true, y_pred)))


#%% 
def train(classifier, X_train=None, y_train=None, X_test=None, y_test=None,\
          score_training_data=False):
    '''Train and score the model'''
    
    #Fitting the model
    classifier.fit(X_train, y_train)
    
    # Model performance for test and train dataset
    if y_test is None or X_test is None:
        print('Skipping the test scores')
        if score_training_data is True:
            model_check(y_train, classifier.predict(X_train))
    elif score_training_data is False:
        model_check(y_test, classifier.predict(X_test))
    else:
        # Score the model on both test and train data
        model_check((y_train, y_test), \
            (classifier.predict(X_train), classifier.predict(X_test)))
    return classifier