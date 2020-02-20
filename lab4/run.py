import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def q1():
    # 1
    data = pd.read_csv('./specs/marks_question1.csv')
    # print(data)
    axe = plt.subplot(1, 1, 1, facecolor='white')
    x = np.arange(0, 12, 1)
    y1 = data.get('midterm')
    y2 = data.get('final')
    axe.set_title('a')
    axe.plot(x, y1)
    axe.plot(x, y2)
    # plt.show()
    # 2
    a = []
    b = []
    for item in y1:
        temp = []
        temp.append(item)
        a.append(temp)

    for item in y2:
        temp = []
        temp.append(item)
        b.append(temp)

    model = LinearRegression()
    model.fit(a, b)
    print("final = " + str(model.coef_) + ' * midterm + ' + str(model.intercept_))
    print(model.predict([[86]]))

q1()
