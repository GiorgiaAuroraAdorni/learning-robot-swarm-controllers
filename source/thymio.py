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

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
        Move the robots in such a way they stand at equal distances from each other.
        Enable communication and send at each timestep a message containing the index.
        It is possible to use the distributed, the omniscient or the learned controller.
        :param dt: control step duration
        """
        self.prox_comm_enable = True
        speed, communication = self.controller.perform_control(self, dt)

        self.motor_left_target = speed
        self.motor_right_target = speed

        self.prox_comm_tx = communication
