
import  configparser
from collections import OrderedDict
from utils.ioUtils import createTemporaryCopy
import subprocess


# Plastimatch
PLASTIMATCH_STAGE = "STAGE"
PLASTIMATCH_GLOBAL = "GLOBAL"
PLASTIMATCH_FIXED = "fixed"
PLASTIMATCH_MOVING = "moving"
PLASTIMATCH_FIXED_MASK = "fixed_mask"
PLASTIMATCH_MOVING_MASK = "moving_mask"
PLASTIMATCH_DVF = "vf_out"
PLASTIMATCH_MOVED = "img_out"
PLASTIMATCH_XF = "xform_out"


def runPlastimatchRegistration(
    plastimatchConfigPath,
    fixedImagePath,
    movingImagePath,
    dvfTargetPath=None,
    movedImagePath=None,
    xfTargetPath=None,
    fixedMaskPath=None,
    movingMaskPath=None,
):

    dictToAdd = {PLASTIMATCH_FIXED: fixedImagePath, PLASTIMATCH_MOVING: movingImagePath}
    if dvfTargetPath is not None:
        dictToAdd[PLASTIMATCH_DVF] = dvfTargetPath
    if movedImagePath is not None:
        dictToAdd[PLASTIMATCH_MOVED] = movedImagePath
    if xfTargetPath is not None:
        dictToAdd[PLASTIMATCH_XF] = xfTargetPath
    if fixedMaskPath is not None:
        dictToAdd[PLASTIMATCH_FIXED_MASK] = fixedMaskPath
    if movingMaskPath is not None:
        dictToAdd[PLASTIMATCH_MOVING_MASK] = movingMaskPath

    dictToAdd = {PLASTIMATCH_GLOBAL: dictToAdd}
    configFilePath = createTemporaryConfigFile(plastimatchConfigPath, dictToAdd)
    print(configFilePath)
    return runSubprocess("plastimatch {}".format(configFilePath), timeout=300)


def createTemporaryConfigFile(plastimatchConfigPath, dictToAdd):
    configFilePath = createTemporaryCopy(plastimatchConfigPath)
    config = readPlastimatchConfig(configFilePath)
    globalSection = [
        sectionName
        for sectionName in config.sections()
        if PLASTIMATCH_GLOBAL in sectionName
    ]
    if len(globalSection) != 1:
        raise ValueError(
            "No or more than one global type of section found in plastimatchFile {}".format(
                plastimatchConfigPath
            )
        )
    globalSection = globalSection[0]

    for section, sectionToAdd in dictToAdd.items():
        if section == PLASTIMATCH_GLOBAL:
            section = globalSection
        for key, value in sectionToAdd.items():
            config.set(section, key, value)

    writePlastimatchConfig(config, configFilePath)
    return configFilePath


def readPlastimatchConfig(filePath):
    config = configparser.ConfigParser(defaults=None, dict_type=multidict, strict=False)
    config.read(filePath)
    return config


def writePlastimatchConfig(config, filePath):
    with open(filePath, "w") as configfile:
        config.write(configfile)
    removeSectionOrdinals(filePath, config)


def removeSectionOrdinals(configFilePath, config):
    with open(configFilePath, "r") as file:
        filedata = file.read()

    for i, section in enumerate(config.sections()):
        filedata = filedata.replace(
            "[{}]".format(section), "[{}]".format(section[: -len(str(i))])
        )

    with open(configFilePath, "w") as file:
        file.write(filedata)


class multidict(OrderedDict):
    _unique = 0  # class variable

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            key += str(self._unique)
            self._unique += 1
        OrderedDict.__setitem__(self, key, val)


def runSubprocess(command, timeout=None):
    print('Running subprocess "{}"'.format(command))
    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        shell=True,
    )
    # TODO: check error throwing or unsuccesfull status
    (output, err) = p.communicate()
    p_status = p.wait(timeout=timeout)
    print("exited with {}".format(p_status))
    return output, p_status
