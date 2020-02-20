import pandas as pd


def run():

    question_2()


def question_1():

    car_data = pd.read_csv('AutoMpg_question1.csv')

    car_data = fill_lost_data(car_data, 'horsepower', find_tag_average(car_data, 'horsepower'))

    car_data = fill_lost_data(car_data, 'origin', find_tag_minimum(car_data, 'origin'))

    car_data.to_csv('./output/question1_out.csv', index=0)


def question_2():

    car_data_a = pd.read_csv('AutoMpg_question2_a.csv')

    car_data_b = pd.read_csv('AutoMpg_question2_b.csv')

    car_data_b.rename(index=str, columns={'name', 'car name'}, inplace=True)

    print(car_data_a)

    print(car_data_b)


# find the average value to fill under a tag.
def find_tag_average(car_data, tag):

    ctr = 0
    average = 0
    for hp in car_data.get(tag):
        if pd.isnull(hp):
            ctr += 1
            # print(ctr)
            continue
        else:
            average += hp

    # print(str(ctr) + ' ' + tag + ' missing detected')

    return average / (len(car_data.get(tag))-ctr)


def find_tag_minimum(car_data, tag):

    first = True
    minimum = 0
    for hp in car_data.get(tag):
        if first and not pd.isnull(hp):
            first = False
            minimum = hp
            continue

        if pd.isnull(hp):
            continue

        else:
            if hp < minimum:
                minimum = hp
                continue

            else:
                continue

    # print('minimum value in ' + tag + ' is: ' + str(minimum))

    return minimum


# fill NaN with a given number
def fill_lost_data(data_set, tag, num):

    data_set[tag] = data_set.get(tag).fillna(num)
    # print(data_set.get(tag).head(20))

    return data_set


def main():

    run()


if __name__ == '__main__':
    main()