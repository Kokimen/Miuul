x = 46

x = 10.3

x = 'hello ai'

type(x)

5 == 4
3 == 3

x = ['a', 'b']

x = {'a': 5, 'b': 6}

x = ('a', 'b')

x = {'a', 'b'}

type(x)

a = 5
b = 6.5

a * 3
a / 7
a * b
a ** 2

int(b)
float(a)

long_str = """three show"""

name = 'John'
name[0]
name[0:3]

"thr" in long_str

dir(int)
dir(str)

len(long_str)

long_str = long_str.upper()

long_str.isupper()

long_str.replace('THR', 'PHR')

long_str.split()

long_str.capitalize()

# listeler değiştirilebilir, sıralıdır, index işlem yapılabilir, kapsayıcıdır.

notes = [1, 2, 3]

names = ['a', 'b', 'c']

not_nam = [1, 2, 3, 'a', 'b', True, [1, 2, 3]]

not_nam[6][0]

len(not_nam)

not_nam.append('c')

not_nam.pop(7)

not_nam.insert(7, 'c')

dictionary = {'year': 2020}

dictionary['year']

'year' in dictionary

dictionary.get('year')

dictionary.keys()

dictionary.values()

dictionary.update({'years': 2022})

# değiştirilemez, sıralı, kapsayıcıdır.

t = ('a', 'b', 5, 6)

t = list(t)

t[0] = 'd'

t = tuple(t)

# değiştirilebilir, sırasız, eşsiz, kapsayıcıdır.

s = {'a', 'b', 5, 6}
s2 = {'a', 'c', 5, 7}

s.difference(s2)  # fark
s - s2

s.symmetric_difference(s2)  # birbirinde olmayanlar

s.intersection(s2)  # kesişim
s & s2

s.union(s2) # birleşim

s.isdisjoint(s2)  # kesişim boş mu?

s.issubset(s2)  # alt küme mi?

s.issuperset(s2)  # kapsıyor mu?

print('a', 'b', sep = '-_-')


def calculate(x):
    print(x * 2)


calculate(10)


def calculate(x, y):
    print(x + y)


calculate(5, 10)


def summer(x, y):
    """
    İki sayıyı topla

    Parameters
    ----------
    x: int ya da float gir
    y: int ya da flaot gir

    Returns
    -------
    int, float toplam değer
    """
    print(x + y)

