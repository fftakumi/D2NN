import numpy as np

def categorical_output(labels, width, padding=0.0, interval=1.0):
    unique = np.sort(np.unique(labels))
    class_num = unique.size
    window_width = (1 - padding) * width / (class_num + (class_num - 1) * interval)
    index = np.repeat(np.arange(width).reshape((1, -1)), repeats=labels.size, axis=0)
    start = np.round(padding * width / 2 + labels * (window_width + window_width * interval))
    end = np.round(start + window_width)
    filters = np.where((start <= index.T) & (index.T <= end), 1, 0)
    return filters.T

