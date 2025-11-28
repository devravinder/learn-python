# Programming Through Python

- concepts for all programs ***
- Syntax differes form PL to PL

---

## data types:-

- how a particual program stores data

Data types:

- Text Based
- Number based
- Yes/No => True/False [ Boolean ]

 eg: delete the data base logs file when it reaches 50 MB. The logs are stored in `msql.log`

- form the above data  
        text data is file name = `msql.log`
        number data is file size = 50

  - using this we can write program to delete the file

  - Psuedo Code:
    - read msql.log file properties
    - if the size is 50 MB or more
      - then delete

---

## Operators
  
Symbols that are used to do some operations/actions

eg:-
   Maths operators: +, -, *, /, %
   Logical: >, <,  and(&), or(|), not(!), equals(=)

- eg:
  - `4 - 2 = 2`  => Valid
    - here `-` is called operator & it does some operation

  - Ravinder - Reddy = Kothabad => Invalid

---

## Conditional Statements

if-else:

- true or false

loops:

- have starting & ending conditions
- multiple

Note:-

- in general laguage, then is if-else
    for each is loop

eg:- a data bases stored 10 days logs in `mssql` folder in 10 log files.
     take backup & delete the log files which are more than 50 MB

Psuedo code:-
   read the mssql folde files
    `for each file`
      get file size
      `check if the file size is 50 MB`
         `then`
         take backup & delete

- in above code `for each` & `if-then` are conditional statements

## Syntax for Python
