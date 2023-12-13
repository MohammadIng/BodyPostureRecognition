import mediapipe as mp
import cv2
import numpy as np
import tools as tl
import time
import Evaluation as ev
import ftools as ft


# Um die Evaluation Aspekte zu betrachten, drücke:
# l: Beleuchtung-Aspekt
# d: Distance-Aspekt
# r: Rotation_Aspekt
# v: Sichtbarkeit-Aspekt

class Finger_Adduction:
    def __init__(self,
                 center=(0, 445),
                 blue=(255, 0, 0),
                 red=(0, 0, 255),
                 green=(0, 255, 0),
                 black=(0, 0, 0),
                 white=(255, 255, 255),
                 mp_drawing=mp.solutions.drawing_utils,
                 mp_hands=mp.solutions.hands,
                 dic={'1-2': [4, 2, 5, 6],
                      '2-3': [8, 5, 9, 12],
                      '3-4': [12, 9, 13, 16],
                      '4-5': [16, 13, 17, 20],
                      'hand_state': [4, 8, 12, 16, 20]}):

        self.blue = blue
        self.red = red
        self.green = green
        self.white = white
        self.black = black
        self.mp_drawing = mp_drawing
        self.mp_hands = mp_hands
        self.dic = dic
        self.center = center

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

    # Methode, um die nächste Bewegung der Hand abhängig von der aktuellen Position der Hand zu bestimmen
    @staticmethod
    def next_move(angles):
        s = 0
        c = 0
        j = 0
        start = len(angles[0]) - 100
        max_values = [30, 15, 15, 15]
        for lists in angles:
            for i in range(start, len(lists)):
                if lists[i] <= 15:
                    s += 1
                elif lists[i] > max_values[j]:
                    c += 1
                if s >= 50:
                    return "spread your Finger"
                if c >= 50:
                    return "close your Finger"
            j += 1

        return " "

    #  Methode, um die Winkel zwischen der Finger zu messen und diese auf dem Bild zu zeichnen
    def draw_finger_angles(self, image, hand, results, joint_list, state, draw, fa):
        angles_list = []
        hand_state = fa.hand_state(hand, state)
        if hand_state[0]:
            i = 0
            for joint in joint_list:
                a = tl.get_points_hand(results)[joint[0]]
                b = tl.get_points_hand(results)[joint[1]]
                c = tl.get_points_hand(results)[joint[2]]
                d = tl.get_points_hand(results)[joint[3]]
                middle_point = tl.mid_2_point(b, c)
                org = tl.mid_2_point(a, d)
                org[0] -= 15
                org[1] += 10

                if joint == joint_list[0]:
                    middle_point[0] -= 7
                    middle_point[1] += 20
                    org = tl.mid_2_point(tl.get_points_hand(results)[4], tl.get_points_hand(results)[6])

                angle = tl.angle_3_points(a, middle_point, d)

                i += 1
                color = self.blue
                if angle <= 0:
                    angle = 0
                    color = self.red

                if draw:
                    cv2.putText(image, str(int(angle)), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
                    cv2.line(image, middle_point, a, self.red, 1)
                    cv2.line(image, middle_point, d, self.blue, 1)
                angles_list.append(int(angle))
        else:
            if draw:
                cv2.putText(image, hand_state[1], self.center, cv2.FONT_HERSHEY_SIMPLEX, 1, self.red, 2, cv2.LINE_AA)
        return angles_list

    # Methode, um das Video zu aufzunehmen und die Hand zu erkennen.
    def finger_adduction(self):

        fa = Finger_Adduction()

        cap = cv2.VideoCapture(0)
        draw = True

        # string = input(str("which Hand R for Right or L for Left "))
        # while string!="R" and string !="L":
        #             string = input(str("wrong input: which Hand R for Right or L for Left "))
        # if string == "R":
        which_Hand = "Right"
        # elif string == "L":
        #     which_Hand = "Left"

        angles_lists = []
        angle0_list = []
        angle1_list = []
        angle2_list = []
        angle3_list = []
        all_frame = 0
        frame_0 = 0

        # mediapipe, um die Hand zu erkenenn, cv2, um das Video aufzunehmen
        with fa.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():

                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                text = ''
                cv2.rectangle(image, (0, 600), (640, 420), fa.white, cv2.FILLED)

                if results.multi_hand_landmarks:

                    frame_0 += 1
                    num = 0
                    for hand in results.multi_hand_landmarks:
                        # fa.mp_drawing.draw_landmarks(image, hand, fa.mp_hands.HAND_CONNECTIONS,fa.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,circle_radius=4),fa.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,circle_radius=2))
                        num += 1
                        if fa.get_label(hand, results):
                            text, coord = fa.get_label(hand, results)
                            cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, fa.white, 2, cv2.LINE_AA)

                    if text == '':
                        text = "Right"
                        coord = tl.get_points_hand(results)[0]
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, fa.white, 2, cv2.LINE_AA)
                    joint_list = [fa.dic['1-2'], fa.dic['2-3'], fa.dic['3-4'], fa.dic['4-5']]
                    angles = fa.draw_finger_angles(image, hand, results, joint_list, text, draw, fa)
                    try:
                        if text == which_Hand:
                            draw = True
                            angle0_list.append(angles[0])
                            angle1_list.append(angles[1])
                            angle2_list.append(angles[2])
                            angle3_list.append(angles[3])



                        else:
                            cv2.putText(image, "Wrong Hand! you have tu show your " + which_Hand + " hand first",
                                        fa.center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, fa.red, 2, cv2.LINE_AA)
                            draw = False

                    except:
                        pass
                    # Winkel zu speichern, um die nächste Bewegung zu bestimmen
                    angles_lists.append(angle0_list)
                    angles_lists.append(angle1_list)
                    angles_lists.append(angle2_list)
                    angles_lists.append(angle3_list)
                    if len(angle0_list) >= 100 and text == which_Hand:
                        message = fa.next_move(angles_lists)
                        cv2.putText(image, message, (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fa.red, 2, cv2.LINE_AA)
                    angles_lists = []

                # Das Video-Fenster zu vergrößern
                frame_resized = tl.rescale_frame(image, scale=1.5)
                cv2.imshow('Finger Adduction', frame_resized)
                all_frame += 1
                print(all_frame)
                if all_frame == 300:
                    print(frame_0)
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == 27: break

            cap.release()
            cv2.destroyAllWindows()
            print(frame_0)


x = Finger_Adduction()
x.finger_adduction()
