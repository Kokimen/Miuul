##############PYTHON#############
# Question One
# Examine whole types of variables.
x = 8

y = 3.2

z = 8j + 18

a = "Hello World"

b = True

c = 23 < 22

l = [1, 2, 3, 4, "String", 3.2, False]

d = {"Name": "Jake",
     "Age": [27, 56],
     "Adress": "Downtown"}

t = ("Machine Learning", "Data Science")

s = {"Python", "Machine Learning", "Data Science", "Python"}

# I don't want to write type for each of variable. So first I make a list which contains every variable.
all_of_them = [x, y, z, a, b, c, l, d, t, s]

# After that I use 'for' loop for every member of the list by list comprehension structures.
[type(col) for col in all_of_them]

# Question Two
# Make all words uppercase, replace comma and dot with space, sparate each word.
text = "The goal is to turn data into information, and information into insight."
# Use 'replace' function to change commas and dots. Uppercase the whole sentence with 'upper' and use 'split' to make
# them show seperately.
text.replace(".", " ").replace(",", " ").upper().split()

# Question Three
lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
# 3.1 Show how many members in the list.
# 'Len' function count members in the list.
len(lst)

# 3.2 Call these indexes; zero and then.
# I use third parameter to get two indexes in one line. Third parameter specifies the step size. Which number of the
# elements to skip.
lst[0:11:10]

# 3.3 Create ["D","A","T","A"] list from the lst.
# Slicing(:) give us the members of the list from zero to four.
lst[0:4]

# 3.4 Delete 8th index member.
# 'Pop' function removes the given index number of members.
lst.pop(8)

# 3.5 Add new member.
# 'Append' function adds new member to last index.
lst.append("NTIST")

# 3.6 Add "N" character to 8th index again.
# 'Insert' adds the character we give to the index we give.
lst.insert(8, "N")

# Question Four
dictionary = {'Christian': ["America", 18],
              'Daisy': ["England", 12],
              'Antonio': ["Spain", 22],
              'Dante': ["Italy", 25]}
# 4.1 Show key variables.
# 'Keys' function shows them.
dictionary.keys()

# 4.2 Show value variables.
# 'Values' funcstion shows them.
dictionary.values()

# 4.3 Update the Daisy number from 12 to 13.
# In dictionary change first index member in Daisy with 13.
dictionary["Daisy"][1] = 13

# 4.4 Add new member in the same context.
# 'Update' function useful for adding new members.
dictionary.update({"Ahmet": ["Turkey", 24]})

# 4.5 Delete Antonio.
# Same as list 'pop' function removes the given key and plus values in the dictionary.
dictionary.pop("Antonio")

# Question Five
# Create function that gets list as argument, seperate single and double numbers as different new list and return them.
number_list = [2, 13, 18, 93, 22]


def number_seperater(number_list):
    # creating two empty list for single & double digits.
    single = []
    double = []
    # making for loop for every number in the list.
    for number in number_list:
        # divide the number two and if remainder is zero, it means the digit is double.
        if number % 2 == 0:
            double.append(number)
        # if remainder is not zero, it means the digit is single.
        else:
            single.append(number)
    # show my list members and return them for using again.
    return single, double


# run the function.
number_seperater(number_list)

# Question Six
# First three students are engineer when other ones are doctor. Ranking them as 1 - 2 - 3 for each job. Use enumerate
# for print the results as 1 - Engineer Max...
students = ["Max", "Mex", "Mix", "Ted", "Tex", "Tec"]
jobs = ["Engineer", "Doctor"]

# 'Enumerate' contains the index number and members.
for index, student in enumerate(students):
    if index < 3:
        # until index is equals the three, print 0th index for jobs and increase index number one by one.
        print(f"{index + 1}, {jobs[0]}, {student}")
        # after index is equals the three, minus two means make index one again and increase for 1st index for jobs.
    else:
        print(f"{index - 2}, {jobs[1]}, {student}")

# Question Seven
# Merge the three list.
lesson_code = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
credit = [3, 4, 2, 4]
quota = [30, 75, 150, 25]

#I use zip for easy to process to for loop. The loop walks through every member of list and print them seperately.
for lesson_code, credit, quota in zip(lesson_code, credit, quota):
    print(f"This is, {lesson_code}, and, credit is, {credit}, for, {quota}, students")


###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])