import mediapipe as mp
import cv2
import numpy as np
import tools as tl
import Evaluation as ev
import time


# Um die Evaluation Aspekte zu betrachten, drücke:
# l: Beleuchtung-Aspekt
# d: Distance-Aspekt
# r: Rotation_Aspekt
# v: Sichtbarkeit-Aspekt(dafür muss man vor dem Ausführen des Programms, die Varible "hands_number" erhöhen)
# Dabei muss man die  Methoden mit den richtigen Parameten aufrufen

class Finger_Flexion:
    def __init__(self,
                 center=(0, 445),
                 blue=(255, 0, 0),
                 red=(0, 0, 255),
                 green=(0, 255, 0),
                 black=(0, 0, 0),
                 white=(255, 255, 255),
                 mp_drawing=mp.solutions.drawing_utils,
                 mp_hands=mp.solutions.hands,
                 dic={
                     '1': [4, 3, 2],
                     '2': [8, 6, 5],
                     '3': [12, 10, 9],
                     '4': [16, 14, 13],
                     '5': [20, 18, 17],
                     'hand_state': [0, 8, 12, 16, 20]},
                 hands_number=1):
        self.blue = blue
        self.red = red
        self.green = green
        self.white = white
        self.black = black
        self.mp_drawing = mp_drawing
        self.mp_hands = mp_hands
        self.dic = dic
        self.center = center
        self.hands_number = hands_number

        # Methode, um zu prüfen, ob die Hand in der richtigen zum Training passenden Position gehalten wird

    def hand_state(self, hand, state):
        joint = self.dic['hand_state']
        b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # 8
        c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # 12
        d = np.array([hand.landmark[joint[3]].x, hand.landmark[joint[3]].y])  # 16
        e = np.array([hand.landmark[joint[4]].x, hand.landmark[joint[4]].y])  # 20

        if b[0] < c[0] < d[0] < e[0] and state == "Right":
            return True, " "
        elif b[0] > c[0] > d[0] > e[0] and state == "Left":
            return True, " "
        return False, "Correct your Hand please"

    # Methode, um zu prüfen, ob die linke Hand oder die rechte Hand gezeigt wird
    def get_label(self, hand, results):
        output = None
        for idx, classification in enumerate(results.multi_handedness):
            if classification.classification[0].index == 0:
                text = classification.classification[0].label
                coords = tuple(np.multiply(
                    np.array(
                        (hand.landmark[self.mp_hands.HandLandmark.WRIST].x,
                         hand.landmark[self.mp_hands.HandLandmark.WRIST].y)),
                    [640, 480]).astype(int))
                output = text, coords
        return output

    #  Methode, um die Winkel zwischen der Finger zu messen und diese auf dem Bild zu zeichnen
    def draw_finger_angles(self, image, hand, results, joint_list, state, draw, fa):
        angles_list = []
        hand_state = fa.hand_state(hand, state)
        if hand_state[0]:
            for joint in joint_list:
                a = tl.get_points_hand(results)[joint[0]]
                b = tl.get_points_hand(results)[joint[1]]
                c = tl.get_points_hand(results)[joint[2]]
                org = a
                angle = tl.angle_3_points(a, b, c)
                color = self.blue
                if angle <= 0:
                    angle = 0
                    color = self.red
                if draw:
                    #   if joint == self.dic['3']:
                    cv2.putText(image, str(int(angle)), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
                    cv2.line(image, a, b, self.red, 1)
                    cv2.line(image, b, c, self.blue, 1)
                    angles_list.append(int(angle))
        else:
            if draw:
                cv2.putText(image, hand_state[1], self.center,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, self.red, 2, cv2.LINE_AA)
        return angles_list, hand_state[0]

    # Methode, um die nächste Bewegung der Hand abhängig von der aktuellen Position der Hand zu bestimmen
    @staticmethod
    def next_move(angles):
        s = 0
        c = 0
        j = 0
        start = len(angles[1]) - 150
        # TODO
        for list in angles:
            for i in range(start, len(list)):
                if list[i] <= 50:
                    s += 1
                elif list[i] > 160:
                    c += 1
                if s >= 50:
                    return "Open your Hand"
                if c >= 50:
                    return "close your Hand"
            j += 1

        return " "

    # Methode, um das Video zu aufzunehmen und die Hand zu erkennen.
    def finger_flexion(self):

        ff = Finger_Flexion()
        cap = cv2.VideoCapture(0)
        draw = False

        string = input(str("which Hand R for Right or L for Left "))
        while string != "R" and string != "L":
            string = input(str("wrong input: which Hand R for Right or L for Left "))
        if string == "R":
            which_Hand = "Right"
        elif string == "L":
            which_Hand = "Left"

        angles_lists = []
        angle0_list = []
        angle1_list = []
        angle2_list = []
        angle3_list = []
        angle4_list = []
        angles_distance_list = []
        rotation_error = []
        visibility = []
        d = False
        l = False
        v = False
        r = False
        all_frame = 0
        frame_0 = 0

        # Methode, um das Video zu aufzunehmen und die Hand zu erkennen.
        with ff.mp_hands.Hands(max_num_hands=ff.hands_number, min_detection_confidence=0.8,
                               min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                text = ''
                cv2.rectangle(image, (0, 600), (640, 420), ff.white, cv2.FILLED)

                if results.multi_hand_landmarks:
                    num = 0
                    for hand in results.multi_hand_landmarks:
                        fa.mp_drawing.draw_landmarks(image, hand, fa.mp_hands.HAND_CONNECTIONS,fa.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,circle_radius=4),fa.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,circle_radius=2))
                        num += 1
                        if ff.get_label(hand, results):
                            text, coord = ff.get_label(hand, results)
                            cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, ff.white, 2, cv2.LINE_AA)

                        # ob die Hand rechts
                    if text == '':
                        text = "Right"
                        coord = tl.get_points_hand(results)[0]
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, ff.white, 2, cv2.LINE_AA)

                    joint_list = [ff.dic['1'], ff.dic['2'], ff.dic['3'], ff.dic['4'], ff.dic['5']]
                    angles, hand_state = ff.draw_finger_angles(image, hand, results, joint_list, text, draw, ff)

                    try:
                        if text == which_Hand:
                            draw = True
                            angle0_list.append(angles[0])
                            angle1_list.append(angles[1])
                            angle2_list.append(angles[2])
                            angle3_list.append(angles[3])
                            angle4_list.append(angles[4])

                            # distance error
                            if d:
                                length = (int(tl.distance(tl.get_points_hand(results)[0],
                                                          tl.get_points_hand(results)[12]) / 10))
                                t = (length, angles[1])
                                angles_distance_list.append(t)
                                print(length)

                            # rotation error
                            if r:
                                rotation_error.append(tl.error_angle("Hand", image, results, True))

                            # visibility_evaluation
                            if v:
                                t = (num, angles[2])
                                visibility.append(t)

                            # light_evaluation
                            if l:
                                frame_0 += 1




                        else:
                            cv2.putText(image, "Wrong Hand! you have tu show your " + which_Hand + " hand first",
                                        ff.center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, ff.red, 2, cv2.LINE_AA)
                            draw = False

                    except:
                        pass

                    # Winkel zu speichern, um die nächste Bewegung zu bestimmen
                    angles_lists.append(angle1_list)
                    angles_lists.append(angle2_list)
                    angles_lists.append(angle3_list)
                    angles_lists.append(angle4_list)
                    if len(angle0_list) >= 150 and text == which_Hand:
                        try:
                            message = ff.next_move(angles_lists)
                            if hand_state:
                                cv2.putText(image, message, (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ff.red, 2,
                                            cv2.LINE_AA)
                        except:
                            pass
                    angles_lists = []
                if l:
                    all_frame += 1

                # Das Video-Fenster zu vergrößern
                frame_resized = tl.rescale_frame(image, scale=1.5)
                cv2.imshow('Finger Adduction', frame_resized)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('l'): l = not (l)
                if key == ord('d'): d = not (d)
                if key == ord('r'): r = not (r)
                if key == ord('v'): v = not (v)
                if key == ord('q') or key == 27: break
            cap.release()
            cv2.destroyAllWindows()

        # distance error
        if d:
            ev.distance_evaluation("Finger_Reflexion/", "Hand", 90, angles_distance_list)

        # rotate error
        if r:
            angles_rotate_list = []
            angles_rotate_list.append(rotation_error)
            angles_rotate_list.append(angle1_list)
            ev.rotation_evaluation("Finger_Flexion/", 180, tl.get_rotate_list(angles_rotate_list))

        # visibility error
        if v:
            ev.visibility_evaluation("visibility_evaluation/", 180, visibility)

        named_tuple = time.localtime()
        t = time.strftime("%m_%d_%Y,%H_%M_%S", named_tuple)

        # plot Data, Daten in Diagramm zu zeigen

        order = int(input("inter 1 to plot Data in Diagram\n"))
        if order == 1:
            angles_lists.append(angle0_list)
            angles_lists.append(angle1_list)
            angles_lists.append(angle2_list)
            angles_lists.append(angle3_list)
            angles_lists.append(angle4_list)
            tl.diagram_line("Finger_Flexion" + t,
                            ["Thump Finger", "Pointer Finger", "Middle Finger", "Ring Finger", "Pinky Finger"],
                            angles_lists, "time", "real angle", "C:/Users/Mohammad/PycharmProjects/System/daten/")

        # Save Data, Daten in excel-Datei zu speichern

        order = int(input("inter 2 to save Data in excel-file\n"))
        if order == 2:
            all_list = []
            angle0_list.insert(0, "Thump:")
            angle1_list.insert(0, "Pointer Finger:")
            angle2_list.insert(0, "Middle Finger:")
            angle3_list.insert(0, "Ring Finger:")
            angle4_list.insert(0, "Pinky Finger:")
            all_list.append(angle0_list)
            all_list.append(angle1_list)
            all_list.append(angle2_list)
            all_list.append(angle3_list)
            all_list.append(angle4_list)
            tl.save_Data("C:/Users/Mohammad/PycharmProjects/System/daten/Finger_Flexion" + t, all_list)
