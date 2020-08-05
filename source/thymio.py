import pyenki


class DistributedThymio2(pyenki.Thymio2):
    """
    Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.

    .. note:: Thymio2 can measure the distances in two ways using the sensors available:

              * ``prox_values​``, an array of 7 values that holds the values of 7 (horizontal) distance sensors around
                its periphery: ​*fll ​(front left left)​, fl ​(front left)​, fc ​(front center)​, fr (front right right)​,
                frr ​(front right)​, bl ​(back left)​, br ​(back right)*.
                The values vary from 0 – when the robot does not see anything – to several thousand – when the robot
                is very close to an obstacle. Thymio updates this array at a frequency of 10 Hz, and generates the
                prox event after every update. The maximum range, in this case, is 14 cm.
              * ``prox_comm_{enable, events, tx}​`` are the horizontal infrared distance sensors to communicate
                value to peer robots within a range of 48 cm. Thymio sends an 11-bit value.
                To use the communication, call the ``​prox_comm_enable(state)`` function, with 1 in a state to enable
                communication or 0 to turn it off. If the communication is enabled, the value in the ``​prox_comm_tx​``
                variable is transmitted every 100 ms. When Thymio receives a value, the ``prox_comm_events`` is fired and
                the value is in the ``prox_comm.rx`` variable.

    :param name: name of the agent
    :param controller: controller to use between OmniscientController, ManualController, LearnedController
    :param index: index of the agents in the row
    :param kwargs: other arguments

    :var initial_position: the initial position of the agent is set to None
    :var goal_position: the goal position of the agent is set to None
    :var goal_angle: the goal angle of the agent is set to None
    :var dictionary: the dictionary containing all the agent attributes is set to None
    :var colour: the colour of the agent is set to None
    :var goal_colour: the goal colour of the agent is set to 'red' or 0,
                        if the agent is in the second half of the row,
                        otherwise is set to 'blue' or 1

    """

    def __init__(self, name, controller, index, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = name
        self.index = index
        self.controller = controller

        self.initial_position = None
        self.goal_position = None
        self.goal_angle = 0

        self.dictionary = None

        self.colour = None

        if self.index > (self.controller.N / 2) - 1:
            self.goal_colour = 0  # red
        else:
            self.goal_colour = 1  # blue

    def colour_thymios(self, dt: float):
        """
        Enable communication and send at each timestep a message decided by the controller.
        Set the top led colour based on the controller decision.
        :param dt: control step duration
        """
        colour, message = self.controller.perform_control(self, dt)

        self.prox_comm_enable = True
        self.prox_comm_tx = message

        self.colour = colour

        if colour == 0:  # red
            self.set_led_top(red=1.0)
        elif colour == 1:  # blue
            self.set_led_top(blue=1.0)
        else:
            self.set_led_top(green=1.0)

    def distribute_thymios(self, dt: float):
        """
        Enable communication and send at each timestep a message decided by the controller.
        Set the velocity of the wheels based on the controller decision.
        :param dt: control step duration
        """
        speed, communication = self.controller.perform_control(self, dt)

        self.prox_comm_enable = True
        self.prox_comm_tx = communication

        self.motor_left_target = speed
        self.motor_right_target = speed

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
                If distribute move the robots in such a way they stand at equal distances from each other.
                If colour color the first half of robot in a different way from the second half.
        :param dt: control step duration
        """

        if self.controller.goal == 'distribute':
            self.distribute_thymios(dt)

        elif self.controller.goal == 'colour':
            self.colour_thymios(dt)

        else:
            raise ValueError("Invalid value for goal!")
