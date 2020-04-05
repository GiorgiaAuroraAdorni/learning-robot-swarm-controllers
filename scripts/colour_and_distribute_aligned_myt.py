import pyenki
import sys
from math import pi
import numpy as np
from pid import PID


# Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
class ColourDistributedThymio2(pyenki.Thymio2):
    INITIAL = 0
    DISTRIBUTING = 1

    def __init__(self, name, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.distributed_controller = PID(-0.01, 0, 0, max_out=500, min_out=-500)
        self.state = self.INITIAL
        self.distribute = True

    def compute_difference(self):
        # Check if there is a robot ahead using the infrared sensor 2 (front-front)
        # If the robot is in the range (~14 cm) the maximum front value should be ~1700
        front = self.prox_values[2]

        # Check if there is a robot ahead using the infrared sensor 5 and 6 (back-left) (back-right)
        back = np.mean(np.array([self.prox_values[5], self.prox_values[6]]))

        # Apply correction to the distance measured by the back sensors
        delta_x = 7.41150769
        x = 7.95
        delta_x_prox_values = 4505 * delta_x / 14
        x_prox_values = 4505 * x / 14
        # print('prox', delta_x_prox_values, x_prox_values)
        correction = x_prox_values - delta_x_prox_values
        # print('correction ', correction)
        # print('back and correction', self.name, back, back + correction)
        back += correction

        return front - back

    def controlStep(self, dt: float) -> None:
        """

        :param dt: control step duration
        """
        #  Move the robots in such a way they stand at equal distances from each other without using communication.
        #  Color the robot half and half they should understand on which side they are compared to the medium.
        if self.state == self.INITIAL:
            # Check if there is a robot ahead using the infrared sensor 2 (front-front)
            front = self.prox_values[2]

            # Check if there is a robot ahead using the infrared sensor 5 and 6 (back-left) (back-right)
            back = np.mean(np.array([self.prox_values[5], self.prox_values[6]]))
            print(self.name, self.position, front, back)

            # if self.name in ['myt1', 'myt10']:
            if front == 0 or back == 0:
                self.distribute = False
                self.set_led_top(red=32)
                # print(self.name, 'blocked')

            self.state = self.DISTRIBUTING

        elif self.state == self.DISTRIBUTING:

            set_point = 0
            # abs_tollerance = 15
            front = self.prox_values[2]

            # Check if there is a robot ahead using the infrared sensor 5 and 6 (back-left) (back-right)
            back = np.mean(np.array([self.prox_values[5], self.prox_values[6]]))

            print(self.name, "F", front, "B", back, "d", self.compute_difference())

            if self.distribute:
                # if not np.isclose(self.compute_difference(), set_point, atol=abs_tollerance):
                    # Aseba uses integers and encodes speed between -500 and +500
                speed = self.distributed_controller.step(self.compute_difference(), dt)

                #     print('Speed', self.name, self.compute_difference(), speed)
                # else:
                #     speed = 0
                #     self.set_led_top(red=32)
                #     print(self.name, "Finished alignment")

                self.motor_left_target = speed
                self.motor_right_target = speed

        # Aseba uses integers and
        # encodes LED values between 0 and 32 (fully lit)
        # self.set_led_top(red=32)


def setup(aseba: bool = False) -> pyenki.World:

    # Create an unbounded world
    world = pyenki.World()

    # Create multiple Thymios and position them such as all x-axes are aligned
    myt_quantity = 8
    myts = [ColourDistributedThymio2(name='myt%d' % (i + 1), use_aseba_units=aseba) for i in range(myt_quantity)]

    # The robots are already arranged in an "indian row" (all x-axes aligned) and within the proximity sensor range
    # ~ 14 cm is the proximity sensors maximal range
    distances = np.random.randint(2, 12, 8)

    # distance between the origin and the front of the robot along x-axis
    constant = 7.95

    for i, myt in enumerate(myts):
        if i == 0:
            prev_pos = 0
        else:
            prev_pos = myts[i - 1].position[0]

        # myt.position = (prev_pos + 5 + constant, 0)
        myt.position = (prev_pos + float(distances[i]) + constant, 0)
        myt.angle = 0
        world.add_object(myt)

    return world


def run(world: pyenki.World, gui: bool = False, T: float = 10, dt: float = 0.1) -> None:

    if gui:
        # We can either run a simulation [in real-time] inside a Qt application
        world.run_in_viewer(cam_position=(70, 0), cam_altitude=150.0, cam_yaw=0.0, cam_pitch=-pi / 2)
    else:
        # or we can write our own loop that run the simulation as fast as possible.
        steps = int(T // dt)
        for _ in range(steps):
            world.step(dt)


if __name__ == '__main__':
    world = setup('--aseba' in sys.argv)

    try:
        run(world, '--gui' in sys.argv)
    except:
        raise
