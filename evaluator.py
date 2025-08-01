from tabulate import tabulate
import copy

class F1Metric:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total = 0
        self.correct = 0

    def add(self, pred, gold):
        if pred and gold:
            self.tp += 1
        elif pred and not gold:
            self.fp += 1
        elif not pred and gold:
            self.fn += 1
        elif not pred and not gold:
            self.tn += 1

        if pred == gold:
            self.correct += 1
        self.total += 1

    def get_metric(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        metric = {
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": self.correct / self.total,
            "Total": int(self.total)
        }
        formatted_data = {key: f"{value:.3f}" if isinstance(value, float) else value for key, value in metric.items()}
        return formatted_data

    
class UnsafeBenchEvaluator:
    def __init__(self):
        # Hate Harassment Violence Self-Harm Sexual Shocking Illegal Activity Deception Political Health Spam
        self.categories = ["Hate", "Harassment", "Violence", "Self-harm", "Sexual", "Shocking",  "Illegal activity", "Deception", "Political","Public and personal health", "Spam", "Overall"]
        cat2metric = {}
        for cat in self.categories:
            cat2metric[cat] = F1Metric()
        self.source2metric = {
            "Laion5B": copy.deepcopy(cat2metric),
            "Lexica": copy.deepcopy(cat2metric),
            "Overall": copy.deepcopy(cat2metric)
        }

    def update(self, preds, gt_dict_list):
        if not isinstance(preds, list):
            preds = [preds]
        for k, v in gt_dict_list.items():
            if not isinstance(v, list):
                gt_dict_list[k] = [v]
        
        for i, pred in enumerate(preds):
            gt_label = 1 if gt_dict_list["safety_label"][i] == "Unsafe" else 0
            gt_cat = gt_dict_list["category"][i]
            gt_source = gt_dict_list["source"][i]
            # Category specific metrics
            self.source2metric[gt_source][gt_cat].add(pred, gt_label)
            self.source2metric["Overall"][gt_cat].add(pred, gt_label)
            # Overall metrics
            self.source2metric[gt_source]["Overall"].add(pred, gt_label)
            self.source2metric["Overall"]["Overall"].add(pred, gt_label)

    def print_table(self, cat2metric):
        table = {
            "Metrics": ["F1", "Precision", "Recall", "Accuracy"]
        }        

        for cat, metric in cat2metric.items():
            cat_metric = metric.get_metric()
            table[cat] = []
            for key in table["Metrics"]:
                table[cat].append(cat_metric[key])

        columns = list(table.keys())
        rows = list(zip(*table.values()))

        # Use tabulate
        print(tabulate(rows, headers=columns, tablefmt='grid'))

    def summarize(self):
        for source, metric in self.source2metric.items():
            print("="*50)
            print(f"Source: {source}")
            self.print_table(metric)
            print("="*50)
        
