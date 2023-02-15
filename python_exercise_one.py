

# ONE#

x = 8
y = 3.2
z = 8j + 18
a = 'hello world'
b = True
c = 23 < 22
l = [1, 2, 3, 4]
d = {'Name': 'Jake',
     'Age': 27,
     'Adress': 'Downtown'}
t = ('Machine Learning', 'Data Science')
s = {'Python', 'Machine Learning', 'Data Science'}

type(x and y and z and l and d and t and s and c and b)

# TWO#
text = 'The goal is to turn data into information, and information into insight.'

print(text.replace(',', ' ').replace('.', ' ').upper().split())

# THREE#
l = ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'N', 'C', 'E']
len(l)
l[0], l[10]
l[0:4]
l.pop(8)
l.append('V')
l.insert(8, 'N')

# FOUR#
dictionary = {'Christian': ['America', 18],
              'Daisy': ['England', 12],
              'Antonio': ['Spain', 22],
              'Dante': ['Italy', 25]}
dictionary.keys()
dictionary.values()
dictionary.update({'Daisy': ['England', 13]})
dictionary['Ahmet'] = ['Turkey', 24]
dictionary.pop('Antonio')

# FIVE#
l = [2, 13, 18, 93, 22]
s = []
d = []


def function():
    for number in l:
        if number % 2 == 0:
            d.append(number)
        else:
            s.append(number)
    return d, s


function()

# SIX#
students = ['Ali', 'Veli', 'Ayşe', 'Talat', 'Zeynep', 'Ece']
departments = ['Mühendislik Fakültesi', 'Tıp Fakültesi']

for index, student in enumerate(students):
    if index < 3:
        print(f'{departments[0]} {index+1}. öğrenci {student}')
    else:
        print(f'{departments[1]} {index-2}. öğrenci {student}')

# SEVEN#
lesson = ['CMP1005', 'PSY1001', 'HUK1005', 'SEN2204']
credit = [3, 4, 2, 4]
limit = [30, 75, 150, 25]

for c, l, o in list(zip(credit, lesson, limit)):
    print(f'Kredisi {c} olan {l} kodlu dersin kontenjanı {o} kişidir')

# EIGHT#

kume1 = {'data', 'python'}
kume2 = {'data', 'function', 'qcut', 'lambda', 'python', 'miuul'}

kume2.issuperset(kume1)
kume1.symmetric_difference(kume2)

