from datetime import date, timedelta


def prev_weekday(date: date) -> date:
    """ Method to get the previous weekday to a provided date """

    date -= timedelta(days=1)
    while date.weekday() > 4:  # Mon-Fri are 0-4
        date -= timedelta(days=1)

    return date
