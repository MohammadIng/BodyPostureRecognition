import time
import cv2
import mediapipe as mp
import tools as tl
import  Evaluation as ev



# Um die Evaluation Aspekte zu betrachten, drücke:
                                                # l: Beleuchtung-Aspekt
                                                # d: Distance-Aspekt
                                                # r: Rotation_Aspekt
# Dabei muss man die  Methoden mit den richtigen Parameten aufrufen

class Elbow_extension():
    def __init__(self,
                 center=(0, 445),
                 blue=(255, 0, 0),
                 red=(0, 0, 255),
                 green=(0, 255, 0),
                 white=(0, 0, 0),
                 black=(255, 255, 255),
                 mp_drawing=mp.solutions.drawing_utils,
                 mp_pose=mp.solutions.pose,
                 dic={
                     'Left_arm': [12, 14, 16],
                     'Right_arm': [11, 13, 15]
                 }
                 ):
        self.blue = blue
        self.red = red
        self.green = green
        self.white = white
        self.black = black
        self.mp_drawing = mp_drawing
        self.mp_pose = mp_pose
        self.dic = dic
        self.center = center

    # Methode, um die nächste Bewegung des Arms abhängig von der aktuellen Position des Arms zu bestimmen
    @staticmethod
    def next_move(angles):
        s = 0;
        c = 0;
        start = len(angles) - 20
        text=" "
        for i in range(start, len(angles)):
            if angles[i] <= 110:
                s += 1
            elif angles[i] > 160:
                c += 1
            if s >= 20:
                text= "Spread your Arm"
            if c >= 20:
                text= "Lower your forearm"

        return text

    #  Methode, um die Winkel zu messen und diese auf dem Bild zu zeichnen
    def draw(self, image, landmarks, which_Arm,angles):
        all_points = tl.get_points_pose(landmarks)

        if which_Arm == "Right":
            list = self.dic['Right_arm']
        else:
            list = self.dic['Left_arm']

        for i in range(3):
            if i + 1 == 3:
                continue
            cv2.line(image, all_points[list[i]], all_points[list[i + 1]], self.red, 5)

        for i in range(3):
           cv2.circle(image, all_points[list[i]], 5, self.blue, -1)

        a = tl.get_points_pose(landmarks)[list[0]]
        b = tl.get_points_pose(landmarks)[list[1]]
        c = tl.get_points_pose(landmarks)[list[2]]
        org = tl.mid_2_point(a, c)
        angle = tl.angle_3_points(a, b, c)
        Cololr = self.blue
        if angle <= 0:
            angle = 0
            Cololr = self.red
        if angle > 180:
            angle = 360 - angle
        cv2.putText(image, str(int(angle)), org, cv2.FONT_HERSHEY_SIMPLEX, 1, Cololr, 2, cv2.LINE_AA)

        cv2.putText(image, str(int(angle)), org,cv2.FONT_HERSHEY_SIMPLEX, 1, Cololr, 2, cv2.LINE_AA)
        message =" "
        n=100
        append=True

        # die nächste Bewegung zu bestimmen
        if not (b[1] in range(a[1] - n, a[1] + n)):
            message = "correct your Arm position"
            append=False
        elif len(angles) > 20:
            message = self.next_move(angles)
        cv2.putText(image, str(message), self.center, cv2.FONT_HERSHEY_SIMPLEX, 1, self.red, 2, cv2.LINE_AA)
        return int(angle),append



    # Methode, um das Video zu aufzunehmen und den Körper zu erkennen.
    def elbow_extension(self):

        cap = cv2.VideoCapture(0)
        e = Elbow_extension()

        string = input(str("which Arm R for Right or L for Left "))
        while string != "R" and string != "L":
           string = input(str("wrong input: which Arm R for Right or L for Left "))
        if string == "R":
               which_Arm = "Right"
        elif string == "L":
               which_Arm = "Left"


        angles = []
        angles_distance_list=[]
        rotation_error=[]
        d=False
        l=True
        r=False
        frame_0=0
        all_frame=0


        # mediapipe, um die Hand zu erkenenn, cv2, um das Video aufzunehmen
        with e.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, (0, 600), (640, 420), e.black, cv2.FILLED)
                e.mp_drawing.draw_landmarks(image, results.pose_landmarks, e.mp_pose.POSE_CONNECTIONS,e.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),e.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                try:
                    landmarks = results.pose_landmarks.landmark
                    angle, append= e.draw(image, landmarks, which_Arm,angles)
                    if append:
                        angles.append(angle)

                        # diatance error
                        if d:
                            length = int(tl.distance(tl.get_points_pose(landmarks)[11], tl.get_points_pose(landmarks)[12]) / 10)
                            t=(length,angle)
                            angles_distance_list.append(t)

                        #rotation error
                        if r:
                            rotation_error.append(tl.error_angle("Pose",image,landmarks,True))

                        # light_evaluation
                        if l:
                            frame_0 += 1


                except:
                   print("No Body")
                if l:
                    all_frame += 1

                # TODO löschen
                if all_frame == 300:
                    break


                # Das Video-Fenster zu vergrößern
                frame_resized = tl.rescale_frame(image, scale=1.5)
                cv2.imshow('Arm Analyse', frame_resized)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('l'): l = not(l)
                if key == ord('d'): d = not (d)
                if key == ord('r'): r = not(r)
                if key == ord('q') or key == 27: break


            cap.release()
            cv2.destroyAllWindows()


            # distance error
            if d:
                ev.distance_evaluation("Elbow_Extension/","Pose",90,angles_distance_list)

            # rotate error
            if r:
                angles_rotate_list=[]
                angles_rotate_list.append(rotation_error)
                angles_rotate_list.append(angles)
                ev.rotation_evaluation("Elbow_extension/",180,tl.get_rotate_list(angles_rotate_list))

            if l:
                print(all_frame)
                print(frame_0)



            named_tuple = time.localtime()
            t = time.strftime("%m_%d_%Y,%H_%M_%S", named_tuple)

            # plot Data, Daten in Diagramm zu zeigen

            order = int(input("inter 1 to plot Data in Diagram\n"))
            if order == 1:
                tl.diagram_line("Elbow_extension" + t,
                                ["Elbow_extension_angle"],
                                [angles], "time", "real angle", "C:/Users/Mohammad/PycharmProjects/System/daten/")

            # Save Data, Daten in excel-Datei zu speichern

            order = int(input("inter 2 to save Data in excel-file\n"))
            if order == 2:
                angles.insert(0,"Elbow_extension_angle")
                tl.save_Data("C:/Users/Mohammad/PycharmProjects/System/daten/Elbow_extension"+t,[angles])