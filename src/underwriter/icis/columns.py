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
    INQ_DATE = 'inq_date'
    CLM_DATE = 'clm_date'
    HOS_SDATE = 'hos_sdate'
    HOS_EDATE = 'hos_edate'
    HOS_DAY = 'hos_day'
    HOS_CNT = 'hos_cnt'
    OUT_CNT = 'out_cnt'
    SUR_CNT = 'sur_cnt'

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
    INQ_DATE = ClaimColumns.INQ_DATE.value
    CLM_DATE = ClaimColumns.CLM_DATE.value
    HOS_SDATE = ClaimColumns.HOS_SDATE.value
    HOS_EDATE = ClaimColumns.HOS_EDATE.value
    HOS_EDATE_MOD = 'hos_edate_mod'
    HOS_DAY = ClaimColumns.HOS_DAY.value
    HOS_CNT = ClaimColumns.HOS_CNT.value
    OUT_CNT = ClaimColumns.OUT_CNT.value
    SUR_CNT = ClaimColumns.SUR_CNT.value
    TYPE = 'type'

# Define column lists using Enum values
CLAIM_COLUMNS: Final[List[str]] = [col.value for col in ClaimColumns]
MAIN_COLUMNS: Final[List[str]] = [col.value for col in MainColumns]
ID_COLUMNS: Final[List[str]] = [col.value for col in ICISColumns]
KCD_COLUMNS: Final[List[str]] = [col.value for col in ClaimColumns if col.name.startswith('KCD_')]
