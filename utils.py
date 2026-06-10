def get_confidence(probability):
    if probability >= 0.8:
        return "High"
    elif probability >= 0.5:
        return "Medium"
    return "Low"