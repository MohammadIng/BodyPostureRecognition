import time
import tools as tl


# distance_evaluation zu analysieren
def distance_evaluation(name, state, real_angle, angles_distance_list):
    dic_hand = {"60": 20, "47": 25, "40": 30, "34": 35, "31": 40, "26": 45, "24": 50, "22": 55, "20": 60, "18": 65,
                "17": 70, "16": 75, "15": 80, "14": 85, "13": 90, "12": 95, "11": 100}
    dic_pose = {"28": 75, "21": 100, "16": 125, "13": 150, "11": 175, "10": 200}
    distance = []
    delta_angles = []
    if state == "Hand":
        dic = dic_hand
    else:
        dic = dic_pose
    real_angles = []
    angles = []
    for tup in angles_distance_list:
        key = str(tup[0])
        if key in dic.keys():
            value = abs(real_angle - tup[1])
            distance.append(dic[key])
            delta_angles.append(value)
            angles.append(tup[1])
            real_angles.append(real_angle)

    tl.digram("distance_error", ["error"], [distance], [delta_angles], "distance", "Deviation of real vlaue", "Null")
    distance.insert(0, "distance")
    real_angles.insert(0, "real value")
    angles.insert(0, "measured angle")
    delta_angles.insert(0, "delta")
    Arrylist = []
    Arrylist.append(distance)
    Arrylist.append(real_angles)
    Arrylist.append(angles)
    Arrylist.append(delta_angles)
    named_tuple = time.localtime()
    t = time.strftime("%m_%d_%Y,%H_%M_%S", named_tuple)
    a = str(real_angle) + "/"
    path = "C:/Users/Mohammad/PycharmProjects/System/Mohammad_distance/";
    tl.save_Data(path + name + a + t, Arrylist)


# rotation_evaluation zu analysieren

def rotation_evaluation(name, real_angle, angles_rotate_list):
    rotate = []
    real_angles = []
    delta_angles = []
    angles = []
    for tup in angles_rotate_list:
        value = abs(real_angle - tup[1])
        rotate.append(tup[0])
        delta_angles.append(value)
        angles.append(tup[1])
        real_angles.append(real_angle)
    tl.digram("rotation_error", ["error"], [rotate], [delta_angles], "distance", "Deviation of real vlaue", "Null")
    rotate.insert(0, "rotate")
    real_angles.insert(0, "real value")
    angles.insert(0, "measured angle")
    delta_angles.insert(0, "delta")
    Arrylist = []
    Arrylist.append(rotate)
    Arrylist.append(real_angles)
    Arrylist.append(angles)
    Arrylist.append(delta_angles)
    named_tuple = time.localtime()
    t = time.strftime("%m_%d_%Y,%H_%M_%S", named_tuple)
    a = str(real_angle) + "/"
    path = "C:/Users/Mohammad/PycharmProjects/System/Mohammad_rotation/";
    tl.save_Data(path + name + a + t, Arrylist)


# visibility_evaluation zu analysieren
def visibility_evaluation(name, real_angle, visibility_liste):
    hand_num = []
    real_angles = []
    delta_angles = []
    angles = []
    for tup in visibility_liste:
        value = abs(real_angle - tup[1])
        hand_num.append(tup[0])
        delta_angles.append(value)
        angles.append(tup[1])
        real_angles.append(real_angle)
    tl.digram("visibility_error", ["error"], [hand_num], [delta_angles], "number of hands", "Deviation of real value",
              "Null")
    hand_num.insert(0, "hand number")
    real_angles.insert(0, "real value")
    angles.insert(0, "measured angle")
    delta_angles.insert(0, "delta")
    Arrylist = []
    Arrylist.append(hand_num)
    Arrylist.append(real_angles)
    Arrylist.append(angles)
    Arrylist.append(delta_angles)
    named_tuple = time.localtime()
    t = time.strftime("%m_%d_%Y,%H_%M_%S", named_tuple)
    a = str(real_angle) + "/"
    path = "C:/Users/Mohammad/PycharmProjects/System/Mohammad_visibility/Hand/";
    tl.save_Data(path + name + a + t, Arrylist)


# Standard Methode, um die Daten bei allen Evaluationsaspekten einzulesen und dazu Diagramme zu erstellen
def evalution():
    p = "C:/Users/Mohammad/Desktop/Daten/"
    name = ["Mohammad/", "Halim/", "Lorka/", "Adnan/"]
    aspect = ["distance/", "rotation/", "visibility/"]
    exercise = ["Finger_Adduction/", "Finger_Flexion/", "Elbow_Extension/"]
    angle = ["0/", "10/", "25/", "36/", "90/", "180/"]
    Y = []
    X = []
    description = ["Mohammad", "Halim", "Lorka", "Adnan"]
    d = []
    try:
        for i in range(2):
            # distance
            path = (p + name[i] + aspect[2] + exercise[1] + angle[5])
            y, x = (tl.read_data_distance_or_visibility("x", "visibility_hand", path, "number of hands", "delta"))
            print(path)
            # rotation
            # path = (p + name[i] + aspect[0] +exercise[0] + angle[1])
            # y,x = (tl.read_data_rotate("Vi",path,"number of hands","delta"))

            Y.append(y)
            d.append(description[i])
            X.append(x)

        tl.digram("Finger_Flexion:visibility_error with real angle 180°", d, X, Y, "number of Hands",
                  "deviation from real angle in Grad °", p + "Figur_15")
        print("true")
    except:
        print("error:-> the Path is not correct")
