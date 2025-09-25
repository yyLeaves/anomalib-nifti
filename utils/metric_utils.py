from anomalib.metrics import Evaluator, AUROC, F1Score

val_metrics = [
    AUROC(fields=("pred_score",  "gt_label"), strict=False, prefix="image_"),
    AUROC(fields=("anomaly_map", "gt_mask"),  strict=False, prefix="pixel_"),
]

test_metrics = [
    AUROC(fields=("pred_score",  "gt_label"), strict=False, prefix="image_"),
    AUROC(fields=("anomaly_map", "gt_mask"),  strict=False, prefix="pixel_"),
    F1Score(fields=("pred_mask", "gt_mask"),  strict=False, prefix="pixel_"),
]

evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)