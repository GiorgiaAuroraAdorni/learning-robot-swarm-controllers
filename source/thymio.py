import pyenki


class DistributedThymio2(pyenki.Thymio2):
    """
    Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
    """

    def __init__(self, name, index, controller, **kwargs) -> None:
        """
        :param name
        :param index
        :param kwargs
        """
        super().__init__(**kwargs)

        self.name = name
        self.index = index

        self.initial_position = None
        self.goal_position = None
        self.goal_angle = 0

        self.dictionary = None

        self.controller = controller
        self.colour = None

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
        Move the robots in such a way they stand at equal distances from each other.
        Enable communication and send at each timestep a message containing the index.
        It is possible to use the distributed, the omniscient or the learned controller.
        :param dt: control step duration
        """

        if self.controller.goal == 'distribute':
            if self.controller.name == "learned":
                self.prox_comm_enable = True
                speed, communication = self.controller.perform_control(self, dt)
                self.prox_comm_tx = communication
            else:
                speed = self.controller.perform_control(self, dt)

            self.motor_left_target = speed
            self.motor_right_target = speed

        elif self.controller.goal == 'colour':
            if self.controller.name == "learned":
                self.prox_comm_enable = True
                colour, communication = self.controller.perform_control(self, dt)
                self.prox_comm_tx = communication
            else:
                colour = self.controller.perform_control(self, dt)

            self.colour = colour
            if colour == 0:  # red
                self.set_led_top(red=1.0)
            elif colour == 1:  # blue
                self.set_led_top(blue=1.0)
            else:
                raise ValueError('Invalid value for colour!')

        else:
            raise ValueError("Invalid value for goal!")
