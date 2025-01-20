from enum import Enum
from typing import List, Final

class ClaimColumns(Enum):
    """
    Columns for the input claim DataFrame.
    Order matches the expected column order in the claim.
    """
    ID = 'id'
    KCD0 = 'kcd0'
    KCD1 = 'kcd1'
    KCD2 = 'kcd2'
    KCD3 = 'kcd3'
    KCD4 = 'kcd4'
    INQUIRY_DATE = 'inq_date'
    CLAIM_DATE = 'clm_date'
    HOS_START_DATE = 'hos_sdate'
    HOS_END_DATE = 'hos_edate'
    HOS_DAY = 'hos_day'
    HOS_COUNT = 'hos_cnt'
    OUT_COUNT = 'out_cnt'
    SUR_COUNT = 'sur_cnt'

class MainColumns(Enum):
    """
    Columns for the main reference DataFrame.
    Contains KCD code mappings and sub-check flags.
    """
    KCD = 'kcd'
    KCD_MAIN = 'kcd_main'
    SUB_CHECK = 'sub_chk'

class ICISColumns(Enum):
    """
    Extended set of columns used in ICIS processing.
    Includes additional columns created during data processing.
    """
    ID = ClaimColumns.ID.value
    INQUIRY_DATE = ClaimColumns.INQUIRY_DATE.value
    CLAIM_DATE = ClaimColumns.CLAIM_DATE.value
    HOS_START_DATE = ClaimColumns.HOS_START_DATE.value
    HOS_END_DATE = ClaimColumns.HOS_END_DATE.value
    HOS_END_DATE_MOD = 'hos_edate_mod'
    HOS_DAY = ClaimColumns.HOS_DAY.value
    HOS_COUNT = ClaimColumns.HOS_COUNT.value
    OUT_COUNT = ClaimColumns.OUT_COUNT.value
    SUR_COUNT = ClaimColumns.SUR_COUNT.value
    TYPE = 'type'

# Define column lists using Enum values
CLAIM_COLS: Final[List[str]] = [col.value for col in ClaimColumns]
MAIN_COLS: Final[List[str]] = [col.value for col in MainColumns]
ID_COLS: Final[List[str]] = [col.value for col in ICISColumns]
KCD_COLS: Final[List[str]] = [col.value for col in ClaimColumns if col.name.startswith('KCD')]
