def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(prediction)):
        if prediction[i] and ground_truth[i]:
            tp += 1
        elif prediction[i] and not ground_truth[i]:
            fp += 1
        elif not prediction[i] and not ground_truth[i]:
            tn += 1
        else:
            fn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp + fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1 = 2*(precision*recall)/(precision+recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    tpn = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            tpn += 1
    return tpn/len(prediction)
