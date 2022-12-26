from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import RobertaForSequenceClassification

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    metric_accuracy = accuracy_score(labels, preds)
    metric_f1_score = f1_score(labels, preds)
    metric_precision_score = precision_score(labels, preds)
    metric_recall_score = recall_score(labels, preds)

    out = {
        "metric_accuracy": metric_accuracy,
        "metric_f1_score": metric_f1_score,
        "metric_precision_score": metric_precision_score,
        "metric_recall_score":metric_recall_score,
    }

    return out

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    return model
