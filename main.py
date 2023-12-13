from Finger_Adduction import Finger_Adduction
from Finger_Flexion import Finger_Flexion
from Elbow_extension import Elbow_extension
import Evaluation as ev


class Run:
    def __init__(self, wich_exercise=0):
        self.which_exercise = wich_exercise

    def main(self):
        x = str(input(
            "wich exercise: \n inter 1 for Finger Adduction\n inter 2 for Finger Flexion\n inter 3 for Elbow extension\n inter 4 for Evaluation\n n=  "));
        # x=2
        r = Run(x)
        r.run(r)

    def run(self, r):
        if self.which_exercise == "1":
            Finger_Adduction.finger_adduction(self)
        elif self.which_exercise == "2":
            Finger_Flexion.finger_flexion(self)
        elif self.which_exercise == "3":
            Elbow_extension.elbow_extension(self)
        elif self.which_exercise == "4":
            ev.evalution()
        else:
            print("you have to type 1, 2, 3 or 4")
            r.main()

        # tl.remove_file("png")


m = Run()
m.main()
