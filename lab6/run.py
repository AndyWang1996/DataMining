import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# plot the points and saving the figure
def show_image(data):
    X = []
    Y = []
    cluster = []
    for x in data['x']:
        X.append(x)
    for y in data['y']:
        Y.append(y)
    for c in data['cluster']:
        cluster.append(c)

    i = 0

    axe = plt.subplot(1, 1, 1, facecolor='white')
    # separate the colors
    while i < len(X):
        if cluster[i] == 0:
            axe.scatter(X[i], Y[i], c='b')
        elif cluster[i] == 1:
            axe.scatter(X[i], Y[i], c='r')
        else:
            axe.scatter(X[i], Y[i], c='g')
        i += 1
    # saving the pictures
    pdf = PdfPages('./output/question_1.pdf')
    pdf.savefig()
    pdf.close()


# show the clusters
def show_clusters(data, label, n):
    result = data[label]
    ctr = 0
    final_group = []

    while ctr < n:
        final_group.append([])
        ctr += 1

    ctr = 0

    for k in result:
        final_group[k].append(ctr)
        ctr += 1

    print(final_group)


def show_image_q3(data, label):
    X = []
    Y = []
    cluster = []
    for x in data['x']:
        X.append(x)
    for y in data['y']:
        Y.append(y)
    for c in data[label]:
        cluster.append(c)

    i = 0

    axe = plt.subplot(1, 1, 1, facecolor='white')
    # separate the colors
    while i < len(X):
        if cluster[i] == 0:
            axe.scatter(X[i], Y[i], c='blue')
        elif cluster[i] == 1:
            axe.scatter(X[i], Y[i], c='red')
        elif cluster[i] == 2:
            axe.scatter(X[i], Y[i], c='green')
        elif cluster[i] == 3:
            axe.scatter(X[i], Y[i], c='black')
        elif cluster[i] == 4:
            axe.scatter(X[i], Y[i], c='yellow')
        elif cluster[i] == 5:
            axe.scatter(X[i], Y[i], c='cyan')
        elif cluster[i] == 6:
            axe.scatter(X[i], Y[i], c='magenta')
        elif cluster[i] == 7:
            axe.scatter(X[i], Y[i], c='firebrick')
        elif cluster[i] == 8:
            axe.scatter(X[i], Y[i], c='tomato')
        elif cluster[i] == 9:
            axe.scatter(X[i], Y[i], c='lightblue')
        elif cluster[i] == 10:
            axe.scatter(X[i], Y[i], c='purple')
        elif cluster[i] == 11:
            axe.scatter(X[i], Y[i], c='pink')
        elif cluster[i] == 12:
            axe.scatter(X[i], Y[i], c='peru')
        elif cluster[i] == 13:
            axe.scatter(X[i], Y[i], c='steelblue')
        elif cluster[i] == 14:
            axe.scatter(X[i], Y[i], c='m')
        elif cluster[i] == 15:
            axe.scatter(X[i], Y[i], c='indigo')
        elif cluster[i] == 16:
            axe.scatter(X[i], Y[i], c='lime')
        elif cluster[i] == -1:
            axe.scatter(X[i], Y[i], c='grey')
        i += 1
    # saving the pictures
    if label == 'kmeans':
        pdf = PdfPages('./specs/output/question_3_1.pdf')
    elif label == 'dbscan1':
        pdf = PdfPages('./specs/output/question_3_2.pdf')
    else:
        pdf = PdfPages('./specs/output/question_3_3.pdf')
    pdf.savefig()
    pdf.close()
    plt.show()
    plt.close()


def q1():
    data = pd.read_csv('./specs/question_1.csv')
    result = KMeans(n_clusters=3, init='k-means++', random_state=0).fit_predict(data)
    data['cluster'] = result
    data.to_csv('./specs/output/question_1.csv', index=False)
    SSE = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(data).inertia_
    print(SSE)
    show_image(data)


def q2():
    data = pd.read_csv('./specs/question_2.csv')
    data.drop(columns=['NAME', 'MANUF', 'TYPE', 'RATING'], inplace=True)
    class1 = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=5)
    class2 = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=100)
    class3 = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=100)

    SSE = []

    result1 = class1.fit_predict(data)
    result1_1 = class1.fit(data)
    result2 = class2.fit_predict(data)
    result2_1 = class1.fit(data)
    result3 = class3.fit_predict(data)

    print(result1)
    print(result1_1.labels_)
    print(result2)
    print(result2_1.labels_)

    data['config1'] = result1
    data['config2'] = result2
    data['config3'] = result3

    show_clusters(data, 'config1', 5)
    show_clusters(data, 'config2', 5)
    show_clusters(data, 'config3', 3)

    SSE.append(class1.fit(data).inertia_)
    SSE.append(class2.fit(data).inertia_)
    SSE.append(class3.fit(data).inertia_)

    print(SSE)

    data.to_csv('./specs/output/question_2.csv', index=False)


def q3():
    data = pd.read_csv('./specs/question_3.csv')
    data.drop(columns='ID', inplace=True)

    class1 = KMeans(n_clusters=7, init='k-means++', max_iter=100, n_init=5, random_state=0)
    result1 = class1.fit_predict(data)

    data['x'] = [((x - min(data['x'])) / (max(data['x']) - min(data['x']))) for x in data['x']]
    data['y'] = [((y - min(data['y'])) / (max(data['y']) - min(data['y']))) for y in data['y']]

    class2 = DBSCAN(eps=0.04, min_samples=4, metric='euclidean', algorithm='auto')
    result2 = class2.fit_predict(data)

    class3 = DBSCAN(eps=0.08, min_samples=4, metric='euclidean', algorithm='auto')
    # class3 = KMeans(n_clusters=7, init='k-means++', max_iter=100, n_init=5, random_state=0)
    result3 = class3.fit_predict(data)

    data['kmeans'] = result1
    data['dbscan1'] = result2
    data['dbscan2'] = result3
    show_image_q3(data, 'kmeans')
    show_image_q3(data, 'dbscan1')
    show_image_q3(data, 'dbscan2')
    data.to_csv('./specs/output/question_3.csv', index=False)
    # print(data)


# q1()
# q2()
q3()
