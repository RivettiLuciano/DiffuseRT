import yaml


DEFAULT_CONFIG_FILE = 'experiments/configDefault.yaml' #TODO: get absolute path

def readConfigAndAddDefaults(configFile,defaultFile = DEFAULT_CONFIG_FILE):
    with open(defaultFile) as file:
        defaultConfig = yaml.load(file, Loader=yaml.FullLoader)

    with open(configFile) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # update default config file with overwritten ones in the config itself
    for key,value in config.items():
        defaultConfig[key] = value

    return defaultConfig