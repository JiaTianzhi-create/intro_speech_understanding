def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.
    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.
    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    month_today, day_today = date
    candidates = []
    for bdate, names in birthdays.items():
        month_b, day_b = bdate
        if (month_b > month_today) or (month_b == month_today and day_b > day_today):
            candidates.append((month_b, day_b, names))
    if not candidates:
        return (1, 1), [] 
    candidates.sort(key=lambda x: (x[0], x[1]))
    next_month, next_day, next_names = candidates[0]
    return (next_month, next_day), next_names
    
