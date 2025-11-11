from torchmetrics.classification import MulticlassAccuracy

def build_metrics(num_classes: int):
    return MulticlassAccuracy(num_classes=num_classes).to("cpu")
