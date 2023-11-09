import yaml
from programs.Aquarium import Aquarium

class Config:
    """
        Class that obtains all the variables from the configuration file
    """
    # Default values
    imageSizeY = 488
    imageSizeX = 648

    PatchyNoise = True
    amountOfData = 50000
    fractionForTraining = .9
    maxAmountOfFishInAllViews = 7

    # TODO: try setting this to a static method to make it more natural
    def __init__(self, pathToYamlFile):
        """
            Essentially just a function to update the variables accordingly
        """
        static_vars = list(vars(Config))[2:-3]

        file = open(pathToYamlFile, 'r')
        config = yaml.safe_load(file)
        keys = config.keys()
        list_of_vars_in_config = list(keys)

        # Updating the static variables
        for var in list_of_vars_in_config:
            if var in static_vars:
                value = config[var]
                line = 'Config.' + var + ' = '
                line += str(value)
                exec(line)
            else:
                print(var + ' is not a valid variable')

        # Writing the variables to the corresponding classes static variables
        Config.set_aquarium_vars()
        Config.set_bounding_box_vars()

    @staticmethod
    def set_aquarium_vars():
        Aquarium.imageSizeX = Config.imageSizeX

        print('setting the vars')

    @staticmethod
    def set_bounding_box_vars():
        print('setting the bounding box vars')

# config = Config('../inputs/config.yaml')