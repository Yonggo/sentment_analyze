def calc_time_to_complete(elapsed_time, current_progress):
    time_kind = "s"
    rest_time = (100 - current_progress) * (elapsed_time / current_progress)
    if rest_time >= 60:
        time_kind = "m"
        rest_time = rest_time / 60.0
    if rest_time >= 60:
        time_kind = "h"
        rest_time = rest_time / 60.0

    x_t = str(rest_time).split(".", 1)[0]
    y_t = str(rest_time).split(".", 1)[1]
    y_t = int(60 * (int(y_t[:2]) / 100)) if len(y_t) > 1 else int(60 * (int(y_t[:1]) / 10))

    if time_kind == "m":
        str_rest_time = "{}m {}s ".format(x_t, y_t)
    elif time_kind == "h":
        str_rest_time = "{}h {}m ".format(x_t, y_t)
    else:
        str_rest_time = "{}s ".format(x_t)

    return str_rest_time


def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", frequency = 10):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        if iteration % frequency == 0:
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r  {prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield i, item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()