# Numbers and Characters Strings#
print("Hello World")
type(9.5)
print(9)

# Assignment and Variables#
a = 9
a
b = "hello ai era"
b
c = 10
a * c
d = a * 5 - c
d
# -------------------------------------------
###VIRTUAL ENVIRONMENT and PACKAGE MANAGEMENT###
# Creating Virtual Environment
# conda
# create - n
# virtual_env

# listing of loaded packages
# conda
# list

# install packages
# conda
# install
# numpy

# install multiple packages
# conda
# install
# numpy
# scipy
# pandas

# delete package
# conda
# remove
# numpy

# install another version of package
# conda
# install
# numpy = 1.20
# .1

# upgrade package
# conda
# upgrade
# numpy

# upgrade all packages
# conda
# upgrade - all
#
# pip: pypi (python package index)

# # install package
# pip
# install
# pandas == 1.5
# .1

# # import list from yaml file
# conda
# env
# create - f
# environment.yaml

# -----------------------------------
# DATA STRUCTURES and QUICK TRAINING
# numbers: integer
# x = 46

# numbers: float
# x = 10.3

# numbers: complex
# x = 2j + 1

# string
# x = "hello ai"

# boolean
# True
# False
# 5 == 4
# 1 == 1

# list
# x = [ "btc", "eth", "xrp" ]
# SQUARE
# PARANTHESES

# dictionary; they are working as two system, name is a key, peter is a value. age is a key, 36 is a value.
# x = {"name": "peter", "age": 36}
# PATTERN
# PARANTHESES and TWO
# DOTS

# tuple
# x = ("btc", "eth", "xrp")
# NORMAL
# PARANTHESES

# set
# x = {"btc", "eth", "xrp"}
# PATTERN
# PARANTHESES

# list, dictionary, tuple and set data structures are also naming as Python Collections (Arrays)

# numbers: int, float, complex
a = 6
b = 10.2
a * 3
a ** 2

# change types
int(b)  # --> float
# to
int

float(a)  # --> int
# to
float

int(a * b / 10)
c = a * b / 10
int(c)

# characters string
print("jon")
print('jon')
name = "jon"

# multiple line strings
long_str = ("Veri Yapıları: Hızlı Özet,\n"
            "Sayılar: int, float, compex")

# reaching the strings members
name[2]

# the strings slice
name[0:2]
long_str[0:10]

# find characters in string
long_str
"Veri" in long_str  # --> python is a
# case
# sensitive
"bool" in long_str

# string methods: class function
dir(str)
# len
a = "cacik"
len(a)
# if the function have meaning in class it is a method. If it is not it is a function. They are the same things.
# the difference is functions are independent when methods are dependent.

# upper and lower
a = "miuul".upper()
a.isupper()

# replace: changes the character
hi = "hello ai era"
hi.replace("l", "p")

# split
"hello ai era".split()

# strip
"ofoff  ".strip()
"ofofo".strip("o")

# capitalize
"foo".capitalize()

# list: more common issue
# değiştirilebilir, sıralıdır, index işlemleri yapılabilir, kapsayıcıdır.
notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v", "d"]
type(names)

not_name = [1, 2, 3, "a", "b", True, [1, 2, 3]]
len(not_name)
not_name[6][1]
notes[0] = 99
notes

not_name[0:4]

# list methods
dir(notes)

len(notes)
len(not_name)

# append: adds member to list
notes.append(100)
notes

# pop: delete with index number
notes.pop(0)

# insert: add with index number
notes.insert(0, 66)

# dictionary
# değiştirilebilir, sırasızdır(3.7 güncellemesinden sonra sıralıdır, kapsayıcıdır.
# key-value

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}
dictionary["REG"]
dictionary = {"REG": ["RMSE", 10],
              "LOG": ["LSE", 20],
              "CART": ["SSE", 30]}

# key finding
"REG" in dictionary

# equals to value to key
dictionary.get("REG")

# change to value
dictionary["REG"] = ["YSA", 10]

# access to all keys and values
dictionary.keys()
dictionary.values()

# convert all couples to tuple list
dictionary.items()

# update the key-value --> we update "REG"
dictionary.update({"REG": "11"})

# add new key-value --> if we use the same key, it will update the key value.
# if it is a different key, it will add for another key-value couple.
dictionary.update({"RF": 10})

# tuples
# değiştirilemez, sıralıdır, kapsayıcıdır. list'in değiştirilemeyen hali. list'e göre daha güvenlidir.
t = ("jo", "ma", 1, 2)
type(t)  # learn type
t[0]  # getting index
t[0:3]  # --> slicing

t = list(t)
t[0] = 99
t = tuple(t)

# set
# değiştirilebilir, sırasızdır ve eşsizdir, kapsayıcıdır.

# difference: difference of two clusters
set1 = {1, 2, 3, 5}
set2 = {1, 2, 3}

# set1 and set2 find different items
set1.difference(set2)
set2.difference(set1)
set1 - set2

# symmetric_difference
set1.symmetric_difference(set2)

# intersection
set1.intersection(set2)
set1 & set2

# union
set1.union(set2)

# isdisjoint: is intersection empty? --> if there is an 'is' in method, waiting for true or false answer.
# according to answer, code system is forwarding
set1.isdisjoint(set2)

# isdissubset: is cluster subset another cluster?
set1.issubset(set2)
set2.issubset(set1)

# issuperset(): does one cluster include another cluster?
set1.issuperset(set2)
set2.issuperset(set1)

# ----------------
###FUNCTIONS###
# function literacy
print("a", "b", sep = "_")


# identify the function --> when we use def, python understand that we are using function.
def calculate(x):
    # --> statement area, what this function do, when we call it?
    print(x * 2)


calculate(5)


# identify function with two parameters
def summer(arg1, arg2):
    print(arg1 + arg2)


summer(8, 9)
summer(arg2 = 5, arg1 = 4)


# docstring --> can add information for people whose reading
def summer(arg1, arg2):
    """
    Sum of two numbers

    Args:
        arg1: int, float
        arg2: int, float

    Returns: int, float
    """
    print(arg1 + arg2)


summer(1, 3)


# function's statement part
# statements (function body)

def say_hi(string):
    print(string)
    print("hi")
    print("hello")

    hi("merhaba")


def multiplication(a, b):
    two_cross_number = a * b
    print(two_cross_number)


multiplication(10, 5)

# the list that can keep the values which inserted
list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 5)
add_element(66, 4)
add_element(5, 4)


# default parameters/arguments
def divide(a, b):
    print(a / b)


divide(1, 2)


def divide(a, b=1):  # -->we can enter default values to beginner people
    print(a / b)


divide(4)


def hi(string="merhaba"):  # --> identify default value as "merhaba", it can work even we don't enter any value.
    print(string)
    print("hi")


hi()

# when to write a function
# stree lamp, warm, moistrue,charge
(56 + 15) / 80
(17 + 45) / 70


def lamp(warm, moisture, charge):
    score = (warm + moisture) / charge
    print(score)


lamp(90, 300, 60)


# return: using function export as an import
def lamp(warm, moisture, charge):
    score = (warm + moisture) / charge
    return score


lamp(98, 12, 78) * 10


def lamp(warm, moisture, charge):
    warm = warm * 2
    moisture = moisture * 2
    charge = charge * 2
    score = (warm + moisture) / charge
    return warm, moisture, charge, score


type(lamp(20, 30, 40))
warm, moisture, charge, score = lamp(20, 30, 40)


# call the function inside of function
def lamp(warm, moisture, charge):
    score = (warm + moisture) / charge
    return score


lamp(98, 12, 78) * 10


def lamp2(x, y):
    return x * 10 / 100 * y * y


lamp2(10, 1)


def all_lamp(warm, moisture, charge, y):
    x = lamp(warm, moisture, charge)
    z = lamp2(x, y)
    print(z * 10)


all_lamp(2, 4, 6, 8)

# local and global variables
list_store = [1, 2]


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 2)

# IF

if 1 == 1:
    print("ekmek")
else:
    print("çay")


def number_check(number):  # birbirini tekrar eden işlemler olursa fonksiyon yaz.
    if number > 10:
        print("yes")
    elif number < 10:
        print("no")
    else:
        print("wow")


number_check(-9)


# ELSE AND ELIF
def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")


number_check(9)
# for loop
students = ['jon', 'jen', 'jun', 'jin']
students[0]

for student in students:
    print(student.upper())

salaries = [100, 200, 300, 400, 500]
for salary in salaries:
    print(int(salary * 1.2))


def new_salary(salary, rate):
    return int(salary * rate)


new_salary(1500, 1.1)

for salary in salaries:
    print(new_salary(salary, 1.1))

salaries2 = [400, 800, 1200, 1600, 2000]

for salary in salaries2:
    print(new_salary(salary, 1.15))

for salary in salaries:
    if salary > 1000:
        print(new_salary(salary, 1.1))
    else:
        print(new_salary(salary, 1.2))

# -----Uygulama Mülakat Sorusu------#
# Aşağıdaki şekilde string değiştiren fonksiyon yazınız
# hi my name is john and i am learning python


range(len('miuul'))
for i in range(len('miuul')):
    print(i)


def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)


alternating("hi my name is john and i am learning python")

##BREAK CONTINUE WHILE##
salaries = [400, 800, 1200, 3000, 2000]
for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

number = 1
while number < 5:
    print(number)
    number += 1

##ENUMERATE##:Üzerinde gezilebilir bir liste içerisinde elemanlara işlem uyghulanırken işlem uygulanan elemanların
# index bilgisini tutan ve daha sonra bu indexe göre işlem yapmamızı sağlayan yapı.

students = ['jon', 'jen', 'jun', 'jin']
for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)
    print(index, student)

# -----Uygulama Mülakat Sorusu------#
# divide students fonksiyon yazınız.
# çift indexte yer alan öğrencileri bir listeye
# tek indexte yer alan öğrencileri başka bir listeye
# bu iki liste tek bir liste olarak return olsun

students = ['jon', 'jen', 'jun', 'jin']


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups


divide_students(students)


##Alternating Fonksiyonunun Enumerate ile Yazımı
def alternating_with_enumerate(string):
    new_string = ""
    for index, letter in enumerate(string):
        if index % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating_with_enumerate("hi my name is john and i am learning python")

##ZIP##!hayat kurtaran bilgi: ayrı listeleri tek bir liste içerisine her birisinde bululnan elemanları aynı sırada
# zipleyerek bir araya getirerek, her birisini tek bir eleman şeklinde görebileceğimiz liste haline getirir.

students = ['jon', 'jen', 'jun', 'jin']
departments = ['math', 'stat', 'phys', 'astro']
ages = [23, 30, 26, 22]

list(zip(students, departments, ages))


# LAMBDA & MAP & FILTER & REDUCE#: vektör seviyesinde işlem yaparlar, lambda önemlidir, apply ile kullanılır.
# lambda: bir kullan at fonksiyondur.

def summer(a, b):
    return a + b


summer(1, 3) * 9

new_sum = lambda a, b: a + b * 9

new_sum(4, 5)

# map#: for yerine kullanılır, döngüden kurtarır.
salaries = [400, 800, 1200, 3000, 2000]


def new_salary(x):
    return x * 1.2


new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))
list(map(lambda x: x * 1.2, salaries))

# filter#: sorgu
list_store = [1, 2, 3, 4, 5, 6]
list(filter(lambda x: x % 2 == 0, list_store))

# reduce#
from functools import reduce

list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)

######LIST COMPREHENSIONS#######: birden fazla kod ve satır ile yapılacak işlemleri tek satırda yapabilen yapılardır, çok önemli!

salaries = [100, 200, 300, 400, 500]


def new_salary(x):
    return x * 1.2


null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

for salary in salaries:
    if salary > 300:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

#eğer if tek başına ise en sağda
#eğer else var ise for en sağda, for'un solunda da else
[salary * 2 if salary > 300 else salary * 1.5 for salary in salaries]
[new_salary(salary) if salary < 300 else salary*0.9 for salary in salaries]

students = ["jen", "jun", "jon", "jin"]
students_no = ["jen", "jon"]

[student.lower() if student in students_no else student.upper() for student in students]
[student.lower() if student not in students_no else student.upper() for student in students]

#########DICT COMPREHENSIONS########
dictionary = {'a': 1, 'b': 2}
dictionary.keys()
dictionary.values()
dictionary.items()

#hayat kurtaran! key ya da valueye özel bir şekilde müdahale etmek
{k: v**2 for(k,v) in dictionary.items()}
{k.upper(): v for (k,v) in dictionary.items()}

#Uygulama Mülakat Sorusu Dict#
#çift sayıların karesi alınacak bir sözlüğe eklenecek
#keyler orijinal kalacak, valueler ise değişecek

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        #hayat kurtaran! elemanları gezilen numbers içerisindeki n'lere dokunmadık, aynı n ifadesinin karesini value
        #bölümüne ekledik, n'i köşeli paranteze alarak değerini key bölümüne ekledik
        new_dict[n] = n**2

{n: n**2 for n in numbers if n % 2 == 0}

##List & Dict Comprehension Uygulamaları# ÖNEMLİDİR
#Bir Veri Steindeki Değişken İsimlerini Değiştirmek#
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

A = []
for col in df.columns:
    A.append(col.upper())
df.columns = A

df.columns = [col.upper() for col in df.columns]

#İsminde INS olan değişkenlerin başına Flag olmayanlarınkine No_Flag eklemek

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

##############################
# BİRLEŞTİRME İŞLEMLERİ (JOIN)#
##############################
# Concat
import numpy as np
import pandas as pd

m = np.random.randint(1, 30, size = (5, 3))
df1 = pd.DataFrame(m, columns = ["var1", "var2", "var3"])  # --> sıfırdan dataframe oluşturur, pd.dataframe
df2 = df1 + 99
pd.concat([df1, df2])
pd.concat([df1, df2], ignore_index = True)

# Merge
df1 = pd.DataFrame({"employees": ["jon", "jan", "jen", "jun"],
                    "group": ["accounting", "engineering", "human resources", "worker"]})
df2 = pd.DataFrame({"employees": ["jon", "jan", "jen", "jun"],
                    "start_date": [2010, 2011, 2012, 2013]})
df3 = pd.merge(df1, df2)
pd.merge(df1, df2, on = "employees")

# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz.
df4 = pd.DataFrame({"group": ["accounting", "engineering", "human resources", "worker"],
                    "manager": ["hayt", "huyt", "hoyt", "heyt"]})
pd.merge(df3, df4)

###########################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN#
###########################################

# MATPLOTLIB
# Kategorik değişken: sütun grafik - countplot ya da bar
# Sayısal değişken: histogram, boxplot

# Kategorik Değişken Görselleştirme
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
# --> kategorik değişkenler ile işlem yapacak olduğumuzda aklımıza gelmesi gereken ilk fonksiyon value_counts
df["sex"].value_counts().plot(kind = "bar")
plt.show()

# Sayısal Değişken Görselleştirme
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

##Matplotlib'in Özellikleri##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Plot Özelliği#
x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()
# Figürleri kapatmadan başka grafik görselleştirme kodu çalıştırılırsa açık olan figür üstüne ekleme yapar.
plt.plot(x, y, "o")
plt.show()

# Marker Özelliği#
y = np.array([13, 28, 11, 100])
plt.plot(y, marker = "o")
plt.show()

# Line Özelliği#
y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle = "dashed", color = "red")
plt.show()

# Multiple Lines#
x = np.array([23, 38, 41, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

# Labels#
x = np.array([23, 38, 41, 90])
y = np.array([13, 28, 11, 100])
plt.plot(x, y)
# Başlık#
plt.title("Bu ana başlık")
# X Eksenini İsimlendirme#
plt.xlabel("X Ekseni İsimlendirmesi")
plt.ylabel("Y Ekseni İsimlendirmesi")
plt.grid()
plt.show()

# Subplots#
x = np.array([23, 38, 41, 90])
y = np.array([13, 28, 11, 100])
plt.subplot(1, 2, 1)  # --> 1e2'lik grafik oluşturuyorum, bu 1. grafik.
plt.title("1")
plt.plot(x, y, marker = "o")
plt.subplot(1, 2, 2)  # --> 1e2'lik grafik oluşturuyorum, bu 2. grafik.
plt.title("2")
plt.plot(x, y, marker = "o")

###SEABORN###
# Kategorik Değişkenler
import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x = df["sex"], data = df)
plt.show()

df['sex'].value_counts().plot(kind = 'bar')
plt.show()

# Sayısal Değişkenler
sns.boxplot(x = df['total_bill'])
plt.show()

df['total_bill'].hist()
plt.show()

######################################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)#
######################################################################
##1. Genel Resim###
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head=5):
    print("### Shape ###")
    print(dataframe.shape)
    print("### Types ###")
    print(dataframe.dtypes)
    print("### Head ###")
    print(dataframe.head())
    print("### Tail ###")
    print(dataframe.tail())
    print("### NA ###")
    print(dataframe.isnull().sum())
    print("### Quantiles ###")
    print(dataframe.describe([0, 0.05, 0.50, 0.95]).T)


check_df(df)

df = sns.load_dataset('tips')

##Kategorik Değişken Analizi (Analysis of Categorical Variables)##
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df['embarked'].value_counts()
df['sex'].unique()
df['sex'].nunique()  # --> kaç adet eşsiz sınıf olduğu bilgisini verir numberunique
df['class'].unique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols += num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

df['survived'].value_counts()
100 * df['survived'].value_counts() / len(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(df)}))


cat_summary(df, 'sex')

for col in cat_cols:
    cat_summary(df, col)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(df)}))

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)


cat_summary(df, 'sex', plot = True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("TİPİ BOOL OLAN GELDİ")
    else:
        cat_summary(df, col, plot = True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
    else:
        cat_summary(df, col, plot = True)

##Sayısal Değişken Analizi##
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float"]]

num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)


num_summary(df, "age", plot = True)

for col in num_cols:
    num_summary(df, col, plot = True)

# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi#
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()
df.info()


# eşsiz değer sayısı 10'dan küçükse kategorik değişken muamelesi yapacağız, eğer 20 den büyükse cardinal değişken
# muamelesi yapacağız docstring, fonksiyona döküman yazmak

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat = cat_cols'un içerisinde
    cat_but_car
    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols += num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float"]]

    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cosl: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(df)}))


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

# bonus
df = sns.load_dataset('titanic')
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int64)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(df)}))

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)


for col in cat_cols:
    cat_summary(df, col, plot = True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)


for col in num_cols:
    num_summary(df, col, plot = True)

# Hedef Değişkenin Kategorik Değişkenler İle Analizi#

cat_summary(df, "survived")
# cinsiyete göre hayatta kalma ortalaması
df.groupby('sex')['survived'].mean()


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({'Target_Mean': dataframe.groupby(categorical_col)[target].mean()}))


target_summary_wtih_cat(df, 'survived', 'pclass')

for col in cat_cols:
    target_summary_with_cat(df, 'survived', col)

# Hedef Değişkenin Sayısal Değişkenler İle Analizi#

df.groupby('survived')['age'].mean()
df.groupby('survived').agg({'age': 'mean'})


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}))


target_summary_with_num(df, 'survived', 'age')

for col in num_cols:
    target_summary_with_num(df, 'survived', col)

# Korelasyon Analizi#
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
# problemli değişkenleri dışarıda bırakır.
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

# korelasyon hesaplama corr
corr = df[num_cols].corr()

sns.set(rc = {'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap = "RdBu")
plt.show()

# Yüksek Korelasyonlu Değişkenlerin Silinmesi

# mutlak değere çevirme
cor_matrix = df.corr().abs()
# korelasyonda aynı şeyleri göstermekten kurtulmak için, hayat kurtaran serisinden ihtiyaç oldukça analizlerde kullanılır
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k = 1).astype(np.bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > .50)]
cor_matrix[drop_list]
df.drop(drop_list, axis = 1)


def high_correlated_cols(dataframe, plot=False, corr_th=.50):
    corr = dataframe.corr()
    cor_matrix = df.corr().abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k = 1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > .50)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc = {'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap = 'RdBu')
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot = True)
df.drop(drop_list, axis = 1)
high_correlated_cols(df.drop(drop_list, axis = 1), plot = True)

# Yaklaşık 600'mblık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
df = pd.read_csv("/Users/Estesia/datasets/train_transaction.csv")
len(df.columns)
df.head

drop_list = high_correlated_cols(df)
len(df.drop(drop_list, axis = 1).columns)
