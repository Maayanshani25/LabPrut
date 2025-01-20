from enum import Enum

class Nodes(Enum):
    WRIST = 0
    MCP1 = 1
    MCP2 = 2
    MCP3 = 3
    MCP4 = 4
    MCP5 = 5
    TIPS1 = 6
    TIPS2 = 7
    TIPS3 = 8
    TIPS4 = 9
    TIPS5 = 10


class FileNames(Enum):
    """
    The mesaurment data files were measured on different days and different tries.
    on the first day there were three tries of control in a row
        and then three tries of HFS in a row.
    on the second day there were two sections of five tries of control in a row 
        and then five tries of HFS in a row.
    
    The data is stored in h5 files.
    """
    CONTROL_DAY1TRY1 =  "data/h5/day1/dataControlDay1Try1.h5"
    CONTROL_DAY1TRY2 =  "data/h5/day1/dataControlDay1Try2.h5"
    CONTROL_DAY1TRY3 =  "data/h5/day1/dataControlDay1Try3.h5"
    CONTROL_DAY2TRY1 =  "data/h5/day2section1/dataControlDay2Try1.h5"
    CONTROL_DAY2TRY2 =  "data/h5/day2section1/dataControlDay2Try2.h5"
    CONTROL_DAY2TRY3 =  "data/h5/day2section1/dataControlDay2Try3.h5"
    CONTROL_DAY2TRY4 =  "data/h5/day2section1/dataControlDay2Try4.h5"
    CONTROL_DAY2TRY5 =  "data/h5/day2section1/dataControlDay2Try5.h5"
    CONTROL_DAY2TRY6 =  "data/h5/day2section2/dataControlDay2Try6.h5"
    CONTROL_DAY2TRY7 =  "data/h5/day2section2/dataControlDay2Try7.h5"
    CONTROL_DAY2TRY8 =  "data/h5/day2section2/dataControlDay2Try8.h5"
    CONTROL_DAY2TRY9 =  "data/h5/day2section2/dataControlDay2Try9.h5"
    CONTROL_DAY2TRY10 = "data/h5/day2section2/dataControlDay2Try10.h5"
    
    HFS_DAY1TRY1 =       "data/h5/day1/dataHFSDay1Try1.h5"
    HFS_DAY1TRY2 =       "data/h5/day1/dataHFSDay1Try2.h5"
    HFS_DAY1TRY3 =       "data/h5/day1/dataHFSDay1Try3.h5"
    HFS_DAY2TRY1 =       "data/h5/day2section1/dataHFSDay2Try1.h5"
    HFS_DAY2TRY2 =       "data/h5/day2section1/dataHFSDay2Try2.h5"
    HFS_DAY2TRY3 =       "data/h5/day2section1/dataHFSDay2Try3.h5"
    HFS_DAY2TRY4 =       "data/h5/day2section1/dataHFSDay2Try4.h5"
    HFS_DAY2TRY5 =       "data/h5/day2section1/dataHFSDay2Try5.h5"
    HFS_DAY2TRY6 =       "data/h5/day2section2/dataHFSDay2Try6.h5"
    HFS_DAY2TRY7 =       "data/h5/day2section2/dataHFSDay2Try7.h5"
    HFS_DAY2TRY8 =       "data/h5/day2section2/dataHFSDay2Try8.h5"
    HFS_DAY2TRY9 =       "data/h5/day2section2/dataHFSDay2Try9.h5"
    HFS_DAY2TRY10 =      "data/h5/day2section2/dataHFSDay2Try10.h5"
    
    RECORDING_FRAMES =   "RecordingFrames.csv"
    