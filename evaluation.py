import torchmetrics.classification as tmc

class MetricFactory:
    def __init__(self, task_type="multilabel", num_labels=1, average="macro"):
        self.task_type = task_type
        self.num_labels = num_labels
        self.average = average

    def get_metrics(self):
        if self.task_type == "binary":
            return {
                "AUROC": tmc.BinaryAUROC(),
                "AUPRC": tmc.BinaryAveragePrecision(),
                "Precision": tmc.BinaryPrecision(),
                "Recall": tmc.BinaryRecall(),
                "F1": tmc.BinaryF1Score(),
                "Accuracy": tmc.BinaryAccuracy(),
                "Specificity": tmc.BinarySpecificity(),
                "Matthews Correlation Coefficient": tmc.BinaryMatthewsCorrCoef(),
            }
        
        elif self.task_type == "multiclass":
            return {
                "AUROC": tmc.MulticlassAUROC(num_classes=self.num_labels, average="macro"),
                "AUROC Macro": tmc.MulticlassAUROC(num_classes=self.num_labels, average="macro"),
                "AUPRC Macro": tmc.MulticlassAveragePrecision(num_classes=self.num_labels, average="macro"),
                "Precision": tmc.MulticlassPrecision(num_classes=self.num_labels, average=self.average),
                "Recall": tmc.MulticlassRecall(num_classes=self.num_labels, average=self.average),
                "F1": tmc.MulticlassF1Score(num_classes=self.num_labels, average=self.average),
                "Matthews Correlation Coefficient": tmc.MulticlassMatthewsCorrCoef(num_classes=self.num_labels),
            }
        
        elif self.task_type == "multilabel":
            return {
                "AUROC": tmc.MultilabelAUROC(num_labels=self.num_labels, average="macro"),
                "AUROC Macro": tmc.MultilabelAUROC(num_labels=self.num_labels, average="macro"),
                "AUROC Micro": tmc.MultilabelAUROC(num_labels=self.num_labels, average="micro"),
                "AUPRC Macro": tmc.MultilabelAveragePrecision(num_labels=self.num_labels, average="macro"),
                "AUPRC Micro": tmc.MultilabelAveragePrecision(num_labels=self.num_labels, average="micro"),
                "Precision": tmc.MultilabelPrecision(num_labels=self.num_labels, average=self.average),
                "Recall": tmc.MultilabelRecall(num_labels=self.num_labels, average=self.average),
                "F1": tmc.MultilabelF1Score(num_labels=self.num_labels, average=self.average),
                "Hamming Loss": tmc.MultilabelHammingDistance(num_labels=self.num_labels),
                "Exact Match Ratio": tmc.MultilabelExactMatch(num_labels=self.num_labels),
            }
        
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
