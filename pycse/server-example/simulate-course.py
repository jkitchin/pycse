import os
import random

letter_grades = ['A++','A+', 'A', 'A-', 'A/B',
                 'B+', 'B', 'B-', 'B/C',
                 'C+', 'C', 'C-', 'C/D',
                 'D+', 'D', 'D-', 'D/R',
                 'R+', 'R', 'R-', 'R--']

COURSE = '06-625'

STUDENTS = ['jim', 'jill', 'bill', 'adam', 'annie', 'john', 'ken','kara']

ASSIGNMENTS = ['1a', '2b', '3a', '5c', '5d', '6e', '7a']

for assignment in ASSIGNMENTS:
    for student in STUDENTS:
        wd = '{0}/{1}'.format(COURSE,
                              assignment)
        if not os.path.isdir(wd):
            os.makedirs(wd)
            
        with open('{0}/{1}/{2}.org'.format(COURSE,
                                           assignment,
                                           student), 'w') as f:
            f.write('#+GRADE: ' + random.choice(letter_grades) + '\n')
            f.write('#+ASSIGNMENT: {0}\n'.format(assignment))
            f.write('#+NAME: {0}\n'.format(student))
                                           
