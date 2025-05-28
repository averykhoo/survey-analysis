import pandas as pd

print('''
################################################
## Question 1: Manipulating Pandas Dataframes ##
################################################
''')

employee_df = pd.DataFrame(
    columns=['id', 'name', 'age', 'address', 'salary'],
    data=[(1, 'Ares', 23, 'Athens', 74635),
          (2, 'Hans', 30, 'Berlin', 72167),
          (3, 'Mary', 24, 'Boston', 75299),
          (4, 'Juan', 43, 'Mexico', 11843),
          (5, 'Dave', 22, 'Sydney', 46681),
          (6, 'Emma', 25, 'London', 43564)])

employee_uin_df = pd.DataFrame(
    columns=['id', 'uin'],
    data=[(1, '57520-0440'),
          (2, '49638-0018'),
          (3, '63550-1941'),
          (4, '68599-6112'),
          (5, '63868-4532'),
          (6, '43198-6341')])

expected_results_df = pd.DataFrame(
    columns=['uin', 'name'],
    data=[('57520-0440', 'Ares'),
          ('63868-4532', 'Dave'),
          ('63550-1941', 'Mary')])

employee_df = pd.DataFrame(
        columns=['id', 'name', 'age', ],
        data=[(1, 'A', 0),
              (2, 'A', 1),
              (3, 'A', 1000000),
              (4, 'B', 23),
              (5, 'B', 24),
              (6, 'B', 25),
              (7, 'B', 26),
              (8, 'B', 27)])

employee_uin_df = pd.DataFrame(
    columns=['id', 'uin'],
    data=[(1, '6'),
          (2, '7'),
          (3, '4'),
          (4, '5'),
          (5, '1'),
          (6, '2'),
          (7, '3'),
          (8, '9')])
expected_results_df = pd.DataFrame(
    columns=['uin', 'name'],
    data=[('6', 'A'),
          ('7', 'A'),
          ('1', 'B'),
          ('5', 'B')])

def process_data(employee_df, employee_uin_df):
    return None  # TODO: fill in your solution here
def process_data(employee_df: pd.DataFrame,
                 employee_uin_df: pd.DataFrame,
                 ) -> pd.DataFrame:
    df = pd.merge(employee_df, employee_uin_df, on='id')
    df = df[df['age'] < 25]
    df = df[['uin', 'name']].sort_values(['name', 'uin'])
    df.reset_index(drop=True, inplace=True)
    return df

your_results_df = process_data(employee_df, employee_uin_df)
passed_q1 = expected_results_df.equals(your_results_df)

print(f'Q1: Expected: \n{expected_results_df}\n')
print(f'Q1: Obtained: \n{your_results_df}\n')
print(f'Q1: Passed: {passed_q1}\n')

print('''
##################################
## Question 2: Technical Coding ##
##################################
''')


def max_operations(nums, k):
    return None  # TODO: fill in your solution here


test_cases = [
    {'nums': [1, 2, 3, 4], 'k': 5, 'result': 2},
    {'nums': [3, 1, 3, 4, 3], 'k': 6, 'result': 1},
    {'nums': [8, 9, 6, 9, 7, 9, 3, 6, 3, 3, 8, 3, 1, 9, 6, 3, 2, 8, 7, 6, 4, 4, 1, 5, 4, 1, 3], 'k': 9, 'result': 9},

    # # hard mode
    # {'nums': list(range(-100, 200, 7)), 'k': 31, 'result': 17},
    # {'nums': list(range(1000)) * 100, 'k': 999, 'result': 50000},
]

passed_q2 = []
for i, test_case in enumerate(test_cases):
    response = max_operations(test_case['nums'], test_case['k'])
    passed = response == test_case['result']
    passed_q2.append(passed)

    print(f'Q2: Test Case {i}: Expected: {test_case["result"]}')
    print(f'Q2: Test Case {i}: Obtained: {response}')
    print(f'Q2: Test Case {i}: Passed: {passed}\n')

# manually check whether all tests passed
if passed_q1 and all(passed_q2):
    print('**********************')
    print('Passed both Q1 and Q2!')
    print('**********************')
