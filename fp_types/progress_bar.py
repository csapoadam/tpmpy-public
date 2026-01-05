# Progress bar for long-running operations
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

from os import linesep
import time

def time_fmt(h, m, s):
    time_str = ''
    if h < 10:
        time_str = time_str + f"0{int(h)}h "
    else:
        time_str = time_str + f"{int(h)}h "

    if m < 10:
        time_str = time_str + f"0{int(m)}m "
    else:
        time_str = time_str + f"{int(m)}m "

    if s < 10:
        time_str = time_str + f"0{int(s)}s"
    else:
        time_str = time_str + f"{int(s)}s"
    return time_str

def get_duration_as_str_from_secs(s):
    hours = s // (60 * 60)
    mins = (s - (hours * 60 * 60)) // 60
    secs = s - (hours * 60 * 60) - (mins * 60)

    return time_fmt(hours, mins, secs)


def progress_bar(current, total, progress_msg=None, bar_length=50, end=None):

    if current == 1 or not hasattr(progress_bar, "start_time"):
        progress_bar.start_time = time.perf_counter()

    fraction = current / total if total > 0 else 1.0
    arrow = int(fraction * bar_length - 1) * '#' + '>'
    padding = (bar_length - len(arrow)) * ' '

    final_end = end if end is not None else linesep
    end_char = final_end if current == total else '\r'

    fraction_str = f" {fraction*100:.2f}% "
    color_code = "\033[32m" if fraction == 1 else "\033[34m"

    elapsed_time = time.perf_counter() - progress_bar.start_time
    estimated_total_time = elapsed_time * (1 / fraction)
    remaining_time = estimated_total_time - elapsed_time

    elapsed_time_str = get_duration_as_str_from_secs(elapsed_time)
    remaining_time_str = get_duration_as_str_from_secs(remaining_time)

    print(f"{color_code}{progress_msg if progress_msg is not None else ""} [{arrow}{padding}] {int(fraction*100)}%\033[0m | Elapsed: {elapsed_time_str} | Remaining: {remaining_time_str}", end=end_char)
