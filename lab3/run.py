import pandas as pd
import numpy as np
import mlxtend.frequent_patterns as mf
from mlxtend.preprocessing import TransactionEncoder
import sklearn


def q1():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Q1_1
    data = pd.read_csv('./specs/gpa_question1.csv')
    count = data.get('count')
    data = data.drop(columns='count')

    # Q1_2
    data = data.apply(handle, axis=1).tolist()
    TE = TransactionEncoder()
    data = TE.fit_transform(data)
    data = pd.DataFrame(data, columns=TE.columns_)
    f_i1 = mf.apriori(data, min_support=0.15, use_colnames=True)
    # f_i1.sort_values(by='support', ascending=False, inplace=True)
    # print(f_i1)

    # Q1_3
    f_i1.to_csv('./specs/output/question1_out_apriori.csv', index=0)

    # Q1_4,5
    rules9 = mf.association_rules(f_i1, metric='confidence', min_threshold=0.9)
    rules9.to_csv('./specs/output/question1_out_rules9.csv', index=0)

    # Q1_6,7
    rules7 = mf.association_rules(f_i1, metric='confidence', min_threshold=0.7)
    rules7.to_csv('./specs/output/question1_out_rules7.csv', index=0)


def q2():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Q2_1
    data = pd.read_csv('./specs/bank_data_question2.csv')
    id = data.get('id')
    data.drop(columns='id', inplace=True)

    # Q2_2
    age = data.get('age')
    income = data.get('income')
    children = data.get('children')

    data.drop(columns='age', inplace=True)
    data.drop(columns='income', inplace=True)
    data.drop(columns='children', inplace=True)

    age = pd.cut(age, 3)
    income = pd.cut(income, 3)
    children = pd.cut(children, 3)

    age = [str(a) for a in age]
    income = [str(a) for a in income]
    children = [str(a) for a in children]

    data['age'] = age
    data['income'] = income
    data['children'] = children
    data['married'] = ['Married_'+str(item) for item in data['married']]
    data['car'] = ['Car_' + str(item) for item in data['car']]
    data['save_act'] = ['Save_Account_' + str(item) for item in data['save_act']]
    data['current_act'] = ['Current_Account_' + str(item) for item in data['current_act']]
    data['mortgage'] = ['Mortgage_' + str(item) for item in data['mortgage']]
    data['pep'] = ['Pep_' + str(item) for item in data['pep']]

    # print(data)

    data = data.apply(handle, axis=1).tolist()
    TE = TransactionEncoder()
    data = TE.fit_transform(data)
    data = pd.DataFrame(data, columns=TE.columns_)

    # Q2_3,4
    f_i2 = mf.fpgrowth(data, min_support=0.2, use_colnames=True)
    f_i2.to_csv('./specs/output/question2_out_fpgrowth.csv', index=0)
    # print(f_i2)

    # Q2_5,6
    rules = mf.association_rules(f_i2, metric='confidence', min_threshold=0.79)
    # print(rules)
    rules.to_csv('./specs/output/question2_out_rules.csv', index=0)

    # Q2_7
    rules.sort_values(by='support', ascending=False, inplace=True)
    # for attribute in data:
    #     print(attribute)

    print(rules)


def handle(data):
    return data.dropna().tolist()


# q1()
q2()
