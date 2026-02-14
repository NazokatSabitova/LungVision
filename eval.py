from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_gen, test_df, labels):

    y_true = test_df[labels].values
    y_pred = model.predict(test_gen)

    # convert probabilities to binary (threshold = 0.5)
    y_pred_bin = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_true.flatten(), y_pred_bin.flatten())
    precision = precision_score(y_true, y_pred_bin, average="macro")
    recall = recall_score(y_true, y_pred_bin, average="macro")
    f1 = f1_score(y_true, y_pred_bin, average="macro")

    return acc, precision, recall, f1
