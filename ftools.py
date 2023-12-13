import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import pathlib
import math as m


def rescale_frame(frame, scale):  # works for image, video, live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def vedio(name):
    cap = cv2.VideoCapture(name)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            text = "Memorize the following Exercise and try to Copy it"
            cv2.putText(frame, text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow(name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def get_pooints_hand(results):
    all_points = []
    for hand in results.multi_hand_landmarks:
        all_points = []
        for joint in range(21):
            a = np.array([hand.landmark[joint].x, hand.landmark[joint].y])  # First coord 8
            point = tuple(np.multiply(a, [640, 480]).astype(int))
            all_points.append(point)
    return all_points


def get_points_pose(landmarks):
    all_points = []
    for joint in range(33):
        a = np.array([landmarks[joint].x, landmarks[joint].y])  # First coord 8
        point = tuple(np.multiply(a, [640, 480]).astype(int))
        all_points.append(point)
    return all_points


def mid_2_point(a, b):
    middle_point = []
    middle_pointx = int((a[0] + b[0]) / 2)
    middle_pointy = int((a[1] + b[1]) / 2)
    middle_point.append(middle_pointx)
    middle_point.append(middle_pointy)
    return middle_point


def angle_3_points(a, b, c):
    z = m.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))  # a b
    y = m.sqrt(pow(b[0] - c[0], 2) + pow(b[1] - c[1], 2))  # a c
    x = m.sqrt(pow(a[0] - c[0], 2) + pow(a[1] - c[1], 2))  # b c
    try:
        alpha = m.acos((pow(z, 2) + pow(y, 2) - pow(x, 2)) / (2 * y * z))
        alpha = np.degrees(alpha)
    except:
        alpha = 0
    return alpha


def x_points_dif(results):
    x1 = get_pooints_hand(results)[9][0]
    x2 = get_pooints_hand(results)[13][0]
    dif = np.abs(x1 - x2)
    return int(dif)


def digramms_4(Arraylist, name):
    index = 2
    titel = ['Thump_Pointer Finger', 'Pointer Finger_Middle Finger', 'Middle Finger-Ring Finger',
             'Ring Finger-Pinky Finger.']
    fig, axs = plt.subplots(index, index, figsize=(11, 25))
    m = 0
    for i in range(index):
        for j in range(index):
            axs[i, j].set_title(titel[m])
            axs[i, j].plot(Arraylist[m])
            m += 1
    axs[0, 0].set_ylabel('angel')
    axs[1, 0].set_ylabel('angel')
    axs[1, 0].set_xlabel('time')
    axs[1, 1].set_xlabel('time')
    plt.savefig(name)
    plt.show()
    plt.close()


def histogramm(Lists):
    zeitArray = []
    val = 0
    for i in range(len(Lists[0])):
        zeitArray.append(val)
        val += 12 / len(Lists[0])

    titel = ["1", "2", "3", "4"]

    # datframe

    data = {"zeit": zeitArray}
    i = 0
    for list in Lists:
        new_data = {titel[i]: list}
        i += 1
        data.update(new_data)

    df = pd.DataFrame(data)
    color = ["yellow", "green", "black", "red"]
    # plot the results
    df.plot(kind='line', x='zeit', y=titel, color=color)
    plt.title("Ellenbogenextension", fontstyle="italic", fontweight="bold", color='blue')
    plt.legend(loc="upper right")
    plt.xlabel("Time", color='blue')
    plt.ylabel("Angles", color='blue')
    plt.show()
    plt.close()


def graphic_error(Arrylist, rotate_Array, Name, x_Achse, y_Achse):
    titel = ['error']
    data = {"rotate": rotate_Array}
    i = 0
    for liste in Arrylist:
        new_data = {titel[i]: liste}
        data.update(new_data)
        i += 1
    df = pd.DataFrame(data)
    df.plot(kind='line', x='rotate', y=['error'], color=['red'])
    plt.title(Name, fontstyle="italic", fontweight="bold", color='blue')
    plt.legend(loc="upper left")
    plt.xlabel(x_Achse, color='blue')
    plt.ylabel(y_Achse, color='blue')
    plt.savefig("Figur 8")
    plt.show()
    plt.close()


def distance(results):
    # 1280 * 720 /  640 * 480
    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
    x1, y1 = get_pooints_hand(results)[5][0] * 2, get_pooints_hand(results)[5][1] * 2
    x2, y2 = get_pooints_hand(results)[17][0] * 1.5, get_pooints_hand(results)[17][1] * 1.5

    distance = int(m.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
    A, B, C = coff
    # print(A, "  ", B, "   ",C )
    distanceCM = A * distance ** 2 + B * distance + C
    return distanceCM


def get_rotate_list(list):
    maxl = max(list[0])
    z = 90 / maxl
    i = 0
    angles_rotate_list = []
    # list.sort()
    while i < (len(list[0])):
        rv = 90 - (list[0][i] * z)
        t = (rv, list[1][i])
        angles_rotate_list.append(t)
        i += 1
    print(angles_rotate_list)
    return angles_rotate_list


def remove_file(end):
    dir_name = pathlib.Path().absolute()
    test = os.listdir(dir_name)
    end = "." + end
    for item in test:
        #  print(item)
        if item.endswith(end):
            os.remove(os.path.join(dir_name, item))


def printinDatei(name, values):
    with open(name + '.csv', 'w', newline='') as file:
        cw = csv.writer(file)
        cw = csv.writer(file, delimiter=';')
        cw.writerows(values)


def dublikat_delet_1Dlist(liste):
    new_liste = []

    for i in liste:
        if i not in new_liste:
            new_liste.append(i)
    return new_liste


def dublikat_delet_2Dlist(Arraylist):
    new_list = []
    for liste in Arraylist:
        liste = dublikat_delet_1Dlist(liste)
        new_list.append(liste)
    return new_list


def close_times(liste):
    clos_liste = []
    for index in range(len(liste)):
        if liste[index] <= 5:
            clos_liste.append(index)
    return clos_liste


def open_times(list, max_val):
    open_liste = []
    for index in range(len(list)):
        if list[index] >= max_val - 10:
            open_liste.append(index)
    return open_liste


def distanz(a, b):
    d = m.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))
    return d


def aver(l, i):
    try:
        s = 0
        for j in range(i - 15, i + 15):
            s += l[j]
        a = s / 30
        print(a)
        return a
    except:
        return 175


def distance_error(list):
    # 20    25   30  35  40  45  50  55  -60-  65  70  75    80  85  90  95  100 d
    # 60    47   40  34  31  26  24  22  -20-  18  17  16    15  14  13  12   11 l
    dis = []
    winkel = []
    print(max(list[1]), " ", min(list[1]))
    dic = {"60": 20, "47": 25, "40": 30, "34": 35, "31": 40, "26": 45, "24": 50, "22": 55, "20": 60, "18": 65, "17": 70,
           "16": 75, "15": 80, "14": 85, "13": 90, "12": 95, "11": 100}
    print(list)
    for i in range(len(list[0])):  # 50
        try:
            print("try")
            v = str(list[0][i])
            print(v)
            if dic[v] not in dis:
                dis.append(dic[v])
                winkel.append(max(list[1]) - aver(list[1], i))
        except:
            pass
    print(dis)
    print(winkel)
    y = []
    y.append(winkel)
    graphic_error(y, dis)


def distance_error_pose(list):
    dic = {"28": 75, "21": 100, "16": 125, "13": 150, "11": 175, "10": 200}
    dis = []
    winkel = []
    for i in range(len(list[0])):  # 50
        try:
            v = str(list[0][i])
            if dic[v] not in dis:
                print(v, " ", dic[v])
                dis.append(dic[v])
                winkel.append(max(list[1]) - aver(list[1], i))
        except:
            pass
    y = []
    y.append(winkel)
    graphic_error(y, dis)


def Finger_Adduction_results(Arraylist):
    length = len(Arraylist[0])
    max_Values = []
    min_Values = []

    for liste in Arraylist:
        max_Values.append(max(liste))  # 4 max values 50 40 30 90
        min_Values.append(min(liste))  # 4 min values 0 0 0 0
    max_Values.insert(0, " maxium: ")
    min_Values.insert(0, " minimum: ")
    return max_Values, min_Values
#  max_min.append(max_Values)
# max_min.append(min_Values)
# printinDatei("Finger_Adduction max and min Vlaues", max_min)


# open = []
# clos = []
# for i in range(4):
#     c = close_times(Arraylist[i])
#     o = open_times(Arraylist[i], max_Values[i])
#     clos.append(c)
#     open.append(o)
# for i in range(4):
#     print(i, "C", clos[i])
#     print(i, "O", open[i])
# count_list = [len(clos[0]), len(clos[1]), len(clos[2]), len(clos[3])]
# for val in range(4):
#     if count_list[val] < 15:
#         print(val, "Not close")
