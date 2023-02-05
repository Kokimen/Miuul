################################
# VERİ YAPILARI, DATA STRUCTURES
################################

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

s.union(s2)  # birleşim

s.isdisjoint(s2)  # kesişim boş mu?

s.issubset(s2)  # alt küme mi?

s.issuperset(s2)  # kapsıyor mu?

##################################################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
##################################################

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
    y: int ya da float gir

    Returns
    -------
    int, float toplam değer
    """
    print(x + y)


list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 8)


def calculate(varm, moisture, charge):
    return (varm + moisture) / charge  # return çıktıyı girdi olarak kullandırtır.


calculate(98, 12, 78) * 10


def standardization(a, b):
    return a * .1 * b * b


standardization(45, 1)


def all_calculation(varm, moisture, charge, c):
    a = calculate(varm, moisture, charge)
    b = standardization(a, c)
    print(b * 10)


all_calculation(1, 3, 5, 12)

names = ["he", "hu", "hi"]

for name in names:
    print(name.upper())

word = "hi my name is john and i am learning python"


def alternating(string):
    new_string = ""
    for words in range(len(string)):
        if words % 2 == 0:
            new_string += string[words].upper()
        else:
            new_string += string[words].lower()
    print(new_string)


alternating(word)

salaries = [1, 2, 3]
for salary in salaries:
    if salary == 2:
        break
    print(salary)

for salary in salaries:
    if salary == 2:
        continue
    print(salary)

for salary in salaries:
    while salary < 5:
        print(salary)
        salary += 2

foods = ['bread', 'meatball', 'soup', 'chicken']

A = []
B = []

for index, food in enumerate(foods):
    if index % 2 == 0:
        A.append(food)
    else:
        B.append(food)

foods = ['bread', 'meatball', 'soup', 'chicken']


def divide_foods(foods):
    groups = [[], []]
    for index, food in enumerate(foods):
        if index % 2 == 0:
            groups[0].append(food)
        else:
            groups[1].append(food)
    print(groups)
    return groups


divide_foods(foods)


def alternating(string):
    new_string = ''
    for index, word in enumerate(string):  # enumerate: range ve len yerine, stringe ve indexe ihtiyaçta kullanılır.
        if index % 2 == 0:
            new_string += word.upper()
        else:
            new_string += word.lower()
    print(new_string)


alternating("i can write some codes thanks to vahit keskin")

names = ['ahmet', 'sevval', 'cansu']
departments = ['muhasebe', 'temizlik', 'tasarim']
ages = [40, 34, 25]

list(zip(names, departments, ages))  # ayrı listeleri tek bir liste icerisinde gösterir.

new_sum = lambda a, b: a + b
new_sum(10, 20)

salaries = [1, 2, 3, 4, 5]


def new_salary(salary):
    return salary * 1.5


for salary in salaries:
    print(new_salary(salary))


list(map(new_salary, salaries))  # map: for döngüsü işlevi görür.

list(map(lambda salary: salary * 1.5, salaries))  # lambda: geçici fonksiyon.

list(filter(lambda salary: salary % 2 == 0, salaries))  # filter: if işlevi görür.

from functools import reduce
reduce(lambda a, b: a + b, salaries)