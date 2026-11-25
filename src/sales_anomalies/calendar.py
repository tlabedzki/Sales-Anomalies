"""
The shop's own calendar of special days: promotions and holidays that every Polish online shop plans for.

The names and dates follow the special days of Data-Generator (settings/commerce_settings.py); the model only learns
from history how strong each day is, it does not read the generator's multipliers.
"""
from __future__ import annotations

import pandas as pd

FIXED_DAYS = {
    "singles_day": ("11-11", "11-11"),
    "valentines_week": ("02-07", "02-13"),
    "mothers_day_week": ("05-19", "05-25"),
    "childrens_day_week": ("05-26", "05-31"),
    "mikolajki": ("12-01", "12-05"),
    "christmas_eve": ("12-24", "12-24"),
    "christmas": ("12-25", "12-26"),
    "new_years_eve": ("12-31", "12-31"),
    "new_year": ("01-01", "01-01"),
}
# First matching name wins when days overlap (the order of Data-Generator).
EVENT_ORDER = ("black_friday", "cyber_monday", "black_week", "singles_day", "valentines_week", "mothers_day_week",
               "childrens_day_week", "mikolajki", "christmas_eve", "christmas", "new_years_eve", "new_year",
               "easter_sunday", "easter_monday")
NO_EVENT = ""


def easter_sunday(year: int) -> pd.Timestamp:
    """Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return pd.Timestamp(year=year, month=month, day=day)


def black_friday(year: int) -> pd.Timestamp:
    """The Friday after the fourth Thursday of November."""
    first = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = first + pd.Timedelta(days=(3 - first.weekday()) % 7)
    return first_thursday + pd.Timedelta(days=22)


def event_days(name: str, year: int) -> list[pd.Timestamp]:
    if name == "black_friday":
        return [black_friday(year)]
    if name == "cyber_monday":
        return [black_friday(year) + pd.Timedelta(days=3)]
    if name == "black_week":
        return [black_friday(year) - pd.Timedelta(days=n) for n in range(1, 5)]
    if name == "easter_sunday":
        return [easter_sunday(year)]
    if name == "easter_monday":
        return [easter_sunday(year) + pd.Timedelta(days=1)]
    start, end = FIXED_DAYS[name]
    return list(pd.date_range(f"{year}-{start}", f"{year}-{end}", freq="D"))


def events(days: pd.DatetimeIndex) -> pd.Series:
    """The special day name of every day ('' for an ordinary day)."""
    names = pd.Series(NO_EVENT, index=days)
    for year in sorted(set(days.year)):
        for name in reversed(EVENT_ORDER):  # earlier names overwrite later ones
            for day in event_days(name, year):
                if day in names.index:
                    names[day] = name
    return names
