def is_unsure(label):
    return label == "Unsure" or label is None

def numerize(label, raise_if_none=False, warn_if_none=True):
    label = clean_label(label)

    if label is None:
        if raise_if_none:
            raise ValueError("Label is None")
        if warn_if_none:
            print("Warning: label is None")
        return 0

    elif label == "n/a":
        return -1

    elif label in ["successful", "yes", "success"]:
        return 1
    elif label in ["unsuccessful", "no", "failure", "unsuccesful"]:
        return 0
    if isinstance(label, int) and label in [0, 1, 2, 3, 4]:
        return label
    elif label.startswith("1.") or "complete failure" in label:
        return 1
    elif label.startswith("2.") or "suboptimal" in label:
        return 2
    elif (
        label.startswith("3.")
        or "somewhat successful" in label
        or "partially successful" in label
    ):
        return 3
    elif label.startswith("4."):
        return 4
    else:
        raise ValueError(f"Unknown label: {label}")


def calculate_unsures(combined_records, cat):
    return int(sum([1 for r in combined_records if is_unsure(r["human"][cat])]))


def clean_label(label):
    if label is None or label == "":
        return None

    if isinstance(label, float):
        return int(label)
    
    if isinstance(label, int):
        return label

    label = label.strip().lower()
    label = label.replace("'", "").replace('"', "")
    return label


def mean(numbers, ndigits=4):
    if len(numbers) == 0:
        return 0

    s = sum(numbers) / len(numbers)
    if ndigits is not None:
        s = round(s, ndigits)
    return s


def harmonic_mean(*numbers):
    if len(numbers) == 0:
        return 0
    for n in numbers:
        if n == 0:
            return 0

    s = len(numbers) / sum([1 / n for n in numbers])

    return s


def accuracy(y_true, y_pred, ndigits=4, percentage=False):
    s = mean([1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)])
    if percentage is True:
        s *= 100
    if ndigits is not None:
        s = round(s, ndigits)
    return s


def precision(y_true, y_pred, ndigits=4, percentage=False):
    tp = sum([1 if yt == yp == 1 else 0 for yt, yp in zip(y_true, y_pred)])
    fp = sum([1 if yt == 0 and yp == 1 else 0 for yt, yp in zip(y_true, y_pred)])
    d = tp + fp
    if d == 0:
        return 0
    s = tp / d
    if percentage is True:
        s *= 100
    if ndigits is not None:
        s = round(s, ndigits)
    return s


def recall(y_true, y_pred, ndigits=4, percentage=False):
    tp = sum([1 if yt == yp == 1 else 0 for yt, yp in zip(y_true, y_pred)])
    fn = sum([1 if yt == 1 and yp == 0 else 0 for yt, yp in zip(y_true, y_pred)])
    if (tp + fn) == 0:
        return 0

    s = tp / (tp + fn)
    if percentage is True:
        s *= 100
    if ndigits is not None:
        s = round(s, ndigits)
    return s


def f1(y_true, y_pred, ndigits=None, percentage=False):
    p = precision(y_true, y_pred, percentage=False)
    r = recall(y_true, y_pred, percentage=False)
    if p + r == 0:
        return 0

    s = 2 * p * r / (p + r)
    if percentage is True:
        s *= 100
    if ndigits is not None:
        s = round(s, ndigits)
    return s


def npv(y_true, y_pred, ndigits=None, percentage=False):
    "negative predictive value"
    tn = sum([1 if yt == yp == 0 else 0 for yt, yp in zip(y_true, y_pred)])
    fn = sum([1 if yt == 1 and yp == 0 else 0 for yt, yp in zip(y_true, y_pred)])
    d = tn + fn
    if tn + fn == 0:
        return 0
    s = tn / (tn + fn)
    if percentage is True:
        s *= 100
    if ndigits is not None:
        s = round(s, ndigits)
    return s


def tpr(y_true, y_pred, ndigits=None, percentage=False):
    "true positive rate"
    return recall(y_true, y_pred, ndigits, percentage)

def tnr(y_true, y_pred, ndigits=None, percentage=False):
    "true negative rate"
    tn = sum([1 if yt == yp == 0 else 0 for yt, yp in zip(y_true, y_pred)])
    n = sum([1 if yt == 0 else 0 for yt in y_true])
    if n == 0:
        return 0
    s = tn / n
    if percentage is True:
        s *= 100
    if ndigits is not None:
        s = round(s, ndigits)
    return s




def more_or_less(yp, ys):
    if yp == ys:
        return 1
    if yp + 1 == ys or yp - 1 == ys:
        return 1
    return 0


def merge_2_and_3(yp, ys):
    if yp == 2 and ys == 3:
        return 1
    if yp == 3 and ys == 2:
        return 1
    return int(yp == ys)


def calculate_agreement_rate(y_prim, y_sec):
    # get agreement
    agreement = {
        k: [int(yp == ys) for yp, ys in zip(y_prim[k], y_sec[k])]
        for k in [
            "trajectory_success",
            "trajectory_side_effect",
            "trajectory_optimality",
            "trajectory_looping",
        ]
    }

    opt = "trajectory_optimality"
    agreement[f"{opt}_approx"] = [
        more_or_less(yp, ys) for yp, ys in zip(y_prim[opt], y_sec[opt])
    ]
    agreement[f"{opt}_diff"] = [
        abs(yp - ys) for yp, ys in zip(y_prim[opt], y_sec[opt])
    ]
    agreement[f"{opt}_merge"] = [
        merge_2_and_3(yp, ys) for yp, ys in zip(y_prim[opt], y_sec[opt])
    ]

    # get agreement rate for each category
    agreement_rate = {k: mean(v) for k, v in agreement.items()}

    return agreement_rate


def get_agreement_scores(y_prim, y_sec):
    agreement = {
        k: {
            "accuracy": accuracy(
                y_prim[k], y_sec[k], ndigits=2, percentage=True
            ),
            "precision": precision(
                y_prim[k], y_sec[k], ndigits=2, percentage=True
            ),
            "recall": recall(
                y_prim[k], y_sec[k], ndigits=2, percentage=True
            ),
            "f1": f1(y_prim[k], y_sec[k], ndigits=2, percentage=True),
            "tnr": tnr(y_prim[k], y_sec[k], ndigits=2, percentage=True),
            "npv": npv(y_prim[k], y_sec[k], ndigits=2, percentage=True),
        }
        for k in [
            "trajectory_success",
            "trajectory_side_effect",
            "trajectory_optimality",
            "trajectory_looping",
        ]
    }

    return agreement


def get_annotator_scores(annotator_pairs, annotator_type):
    y = {
        k: [
            numerize(v[annotator_type]["human"][k])
            for k2, v in annotator_pairs.items()
            if not is_unsure(v["secondary"]["human"][k])
            and not is_unsure(v["primary"]["human"][k])
        ]
        for k in [
            "trajectory_success",
            "trajectory_side_effect",
            "trajectory_optimality",
            "trajectory_looping",
        ]
    }
    return y