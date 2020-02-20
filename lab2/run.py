import pandas as pd
import numpy as np
import nltk
from sklearn.decomposition import PCA


def run():

    bot()

    # q1()

    # q2()


def bot():
    ori = ['m e a t ', 'v e g e t a b l e ']
    text1 = []
    for x in ori:
        text1.append('n o - ' + x)
        text1.append('n o ' + x)
        text1.append(x + 'f r e e ')
        text1.append('d i s ' + x)
        text1.append('u n ' + x)

    text = []
    for x in text1:
        text.append(x + 'c h i n e s e t a k e a w a y ')
    for x in text1:
        text.append(x + 'b u r g e r ')
    for x in text1:
        text.append(x + 'p i z z a ')

    for t in text:
        temp = t.split(" ")
        print("\"", end="")
        for c in temp:
            print("["+c.upper()+"|"+c.lower()+"]", end="")
        print("\"")


def q2():

    # Q2(1)
    data = pd.read_csv('./specs/DNAData_question2.csv')
    pca = PCA(n_components=0.95)  # with 95.2% (22)
    new_data = pca.fit_transform(data)
    # print(np.sum(pca.explained_variance_ratio_))  # with 95.2%

    # Q2(2)
    r_data = pd.DataFrame(new_data)

    index = 0

    for column in r_data:
        temp = pd.cut(r_data[index], 10)
        attribute_tag = 'pca' + str(index) + '_width'
        data[attribute_tag] = temp
        index += 1

    # Q2(3)

    index = 0

    for column in r_data:
        temp = pd.qcut(r_data[index], q=10)
        attribute_tag = 'pca' + str(index) + '_freq'
        data[attribute_tag] = temp
        index += 1

    # Q2(4)
    data.to_csv('./output/question2_out.csv', index=0)
    # print(data)


def q1():
    data = pd.read_csv('./specs/SensorData_question1.csv')

    # Q1(1)
    ori_input12 = find_attribute_with_tag(data, "Input12")
    out_3 = []
    ori_input3 = find_attribute_with_tag(data, "Input3")
    out_12 = []
    for num in ori_input3:
        out_3.append(round(num, 3))

    for num in ori_input12:
        out_12.append(round(num, 3))

    data['Original Input3'] = out_3
    data['Original Input12'] = out_12

    # Q1(2)
    data['Input3'] = z_normalized(data, 'Input3')

    # Q1(3)
    data['Input12'] = min_max_normalized(data, 'Input12')

    # Q1(4)
    data['Average Input'] = get_average_input(data)

    # Q1(5)
    data.to_csv('./output/question1_out.csv', index=0)


def get_average_input(data):

    result = []
    index = 0

    while index is not 199:
        result.append(
            (data['Input1'][index] +
             data['Input2'][index] +
             data['Input3'][index] +
             data['Input4'][index] +
             data['Input5'][index] +
             data['Input6'][index] +
             data['Input7'][index] +
             data['Input8'][index] +
             data['Input9'][index] +
             data['Input10'][index] +
             data['Input11'][index] +
             data['Input12'][index])/12)
        index += 1
    return result


def min_max_normalized(data, tag):

    result = []

    min_v = pd.Series.min(data[tag])
    max_v = pd.Series.max(data[tag])

    for value in data[tag]:
        result.append((value-min_v)/(max_v-min_v))

    return result


def z_normalized(data, tag):

    result = []

    mean = pd.Series.mean(data[tag])
    std = pd.Series.std(data[tag])

    for value in data[tag]:
        result.append((value-mean)/std)

    return result


def find_attribute_with_tag(data_set, tag):
    return data_set.get(tag)


def main():

    run()


if __name__ == '__main__':

    main()
