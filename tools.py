import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import math as m
import pathlib
import pandas as pd
import os


# Video-Fenster zu vergrößern
def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# Umwandeln die x und y Koordinaten von Hand von Pixel in mm(Millimeter)
def get_points_hand(results):
    all_points = []
    for hand in results.multi_hand_landmarks:
        all_points = []
        for joint in range(21):
            a = np.array([hand.landmark[joint].x, hand.landmark[joint].y])  # First coord 8
            point = tuple(np.multiply(a, [640, 480]).astype(int))
            all_points.append(point)
    return all_points


# Umwandeln die x und y Koordinaten vom Körper von Pixel in mm(Millimeter)
def get_points_pose(landmarks):
    all_points = []
    for joint in range(33):
        a = np.array([landmarks[joint].x, landmarks[joint].y])  # First coord 8
        point = tuple(np.multiply(a, [640, 480]).astype(int))
        all_points.append(point)
    return all_points


# den Punke zwischen zwei Punkten zu berechnen
def mid_2_point(a, b):
    middle_point = []
    middle_pointx = int((a[0] + b[0]) / 2)
    middle_pointy = int((a[1] + b[1]) / 2)
    middle_point.append(middle_pointx)
    middle_point.append(middle_pointy)
    return middle_point


# die Winkel zwischen drei Punkten zu berechnen
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


# die Differenz von den X-Koordinaten zwei Punkten zu berechnen
def x_points_dif(results):
    x1 = get_points_hand(results)[9][0]
    x2 = get_points_hand(results)[13][0]
    dif = np.abs(x1 - x2)
    return int(dif)


# Daten in excel Datei zu speichern
def save_Data(name, values):
    with open(name + '.csv', 'w', newline='') as file:
        cw = csv.writer(file)
        cw = csv.writer(file, delimiter=';')
        cw.writerows(values)


# Daten in Digaramm zu zeihnen
def digram(name, tit, y_lists, x_lists, x_Achse_t, y_Achse_t, save_path):
    m = ["o", ">", "<", "x"]
    color = ['blue', 'red', 'green', 'yellow', 'black']
    for i in range(len(y_lists)):
        plt.scatter(y_lists[i], x_lists[i], c=color[i], label=tit[i], marker=m[i])
    plt.title(name, fontstyle="italic", fontweight="bold", color='blue')
    plt.legend(loc="best")
    plt.xlabel(x_Achse_t)
    plt.ylabel(y_Achse_t)
    if len(y_lists) >= 2:
        plt.savefig(save_path)
    plt.show()


# Daten in Digaramm zu zeihnen
def diagram_line(name, titel, y_lists, x_Achse_t, y_Achse_t, save_path):
    x_list = []
    val = 0
    for i in range(len(y_lists[0])):
        x_list.append(val)
        val += len(y_lists[0]) / len(y_lists[0])

    color = ["red", "green", "black", "yellow", "blue", "pink"]
    c = []
    data = {"x_list": x_list}
    i = 0
    for list in y_lists:
        new_data = {titel[i]: list}
        c.append(color[i])
        i += 1
        data.update(new_data)

    df = pd.DataFrame(data)
    df.plot(kind='line', x='x_list', y=titel, color=color)
    plt.title(name, fontstyle="italic", fontweight="bold", color='blue')
    plt.legend(loc="best")
    plt.xlabel(x_Achse_t, color='blue')
    plt.ylabel(y_Achse_t, color='blue')
    plt.savefig(save_path + name)
    plt.show()
    plt.close()


# Rotationswinkel beim Rotationsfehler zu generieren
def get_rotate_list(list):
    maxl = max(list[0])
    z = 90 / maxl
    i = 0
    angles_rotate_list = []
    while i < (len(list[0])):
        rv = int(90 - (list[0][i] * z))
        t = (rv, list[1][i])
        angles_rotate_list.append(t)
        i += 1
    print(angles_rotate_list)
    return angles_rotate_list


# Abstand zwischen zwei Punkten zu berechnen
def distance(a, b):
    d = m.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))
    return d


# Hilf_Winkel zu berechnen, um die Rotationsgrad zu generieren
def error_angle(state, image, results, draw):
    if state == "Hand":
        a = get_points_hand(results)[17]
        b = get_points_hand(results)[0]
        c = get_points_hand(results)[5]
    elif state == "Pose":
        a = get_points_pose(results)[11]
        b = get_points_pose(results)[24]
        c = get_points_pose(results)[12]
    init_angle = int(angle_3_points(a, b, c))

    if draw:
        cv2.line(image, a, b, (0, 0, 0), 1)
        cv2.line(image, b, c, (0, 0, 0), 1)
        cv2.circle(image, b, 5, (255, 0, 0), -1)
        cv2.putText(image, str(init_angle), b, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return (init_angle)


# Daten von Exel-Datein bei Rotation_Evaluation_Aspekt einzulesen
def read_data_rotate(name_of_graph, path, x_achse, y_achse):
    rotate_grad = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    dic = {}
    for i in range(len(rotate_grad) - 1):
        k = str(rotate_grad[i]) + "_" + str(rotate_grad[i + 1])
        w = [rotate_grad[i], rotate_grad[i + 1], 0, 0, 0]
        i = {k: w}
        dic.update(i)

    dir_name = pathlib.Path(path)
    names = os.listdir(dir_name)
    l = []
    l1 = []
    rotate = []
    delta = []
    for name in names:
        if name.endswith(".csv"):
            d1 = pd.read_csv(path + name, sep=";", decimal=",", skipfooter=3, engine='python')
            d2 = pd.read_csv(path + name, sep=";", decimal=",", skiprows=3, engine='python')
            l1.append(d1)
            l1.append(d2)

    for i in range(len(l1)):
        if i % 2 == 0:
            for val in l1[i]:
                if val != "rotate":
                    rotate.append(int(float(val)))
        else:
            for val in l1[i]:
                if val != "delta":
                    delta.append(int(float(val)))

    for i in range(len(rotate)):
        t = (rotate[i], delta[i])
        l.append(t)

    for tup in l:
        for val in dic.values():
            if tup[0] >= val[0] and tup[0] < val[1] and val[0] != val[1]:
                k = str(val[0]) + "_" + str(val[1])
                dic[k][2] += tup[1]
                dic[k][3] += 1

    for key in dic.keys():
        try:
            dic[key][4] = (dic[key][2] / dic[key][3])
        except:
            pass
    y_list = []
    x_list = []
    error = []
    rotate = []
    i = 0
    for key in dic.keys():
        rotate.append(str(rotate_grad[i]))
        i += 1
        error.append(dic[key][4])
    y_list.append(error)
    x_list.append(rotate)
    digram(name_of_graph, ["rotate"], x_list, y_list, x_achse, y_achse, " ")
    return error, rotate


# Daten von Exel-Datein bei distance_or_visibility_Evaluation_Aspekt einzulesen
def read_data_distance_or_visibility(name_of_graph, state, path, x_achse, y_achse):
    dic = {}
    if state == "visibility_hand":
        lebel = state
        dic = {"1": [0, 0, 0], "2": [0, 0, 0], "3": [0, 0, 0], "4": [0, 0, 0]}
    else:
        if state == "Hand":
            lebel = "Hand distance"
            d = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        elif state == "Pose":
            lebel = "Pose distance"
            d = [75, 100, 125, 150, 175, 200]

        for val in d:
            k = str(val)
            w = [0, 0, 0]
            i = {k: w}
            dic.update(i)
    dir_name = pathlib.Path(path)
    names = os.listdir(dir_name)
    l = []
    l1 = []
    distance_hand_number = []
    delta = []
    for name in names:
        if name.endswith(".csv"):
            d1 = pd.read_csv(path + name, sep=";", decimal=",", skipfooter=3, engine='python')
            d2 = pd.read_csv(path + name, sep=";", decimal=",", skiprows=3, engine='python')
            l1.append(d1)
            l1.append(d2)

    for i in range(len(l1)):
        if i % 2 == 0:
            for val in l1[i]:
                if val != "distance" and val != "hand number":
                    try:
                        distance_hand_number.append(int(float(val)))
                    except:
                        print("error")
        else:
            for val in l1[i]:
                if val != "delta":
                    delta.append(int(float(val)))

    for i in range(len(distance_hand_number)):
        t = (distance_hand_number[i], delta[i])
        l.append(t)

    for tup in l:
        k = str(tup[0])
        dic[k][0] += tup[1]
        dic[k][1] += 1
    for key in dic.keys():
        try:
            dic[key][2] = (dic[key][0] / dic[key][1])
        except:
            pass
    y_list = []
    x_list = []
    error = []
    distance_hand_number = []
    for key in dic.keys():
        distance_hand_number.append(int(key))
        error.append(dic[key][2])
    y_list.append(error)
    y_list.append(error)
    x_list.append(distance_hand_number)
    # digram(name_of_graph,[lebel],x_list, y_list,x_achse, y_achse," ")
    return error, distance_hand_number


# File zu löschen
def remove_file(end):
    dir_name = pathlib.Path().absolute()
    test = os.listdir(dir_name)
    end = "." + end
    for item in test:
        if item.endswith(end):
            os.remove(os.path.join(dir_name, item))
