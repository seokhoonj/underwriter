from .columns import ID_COLS, KCD_COLS, ClaimColumns, MainColumns
from ..utils.helpers import fill_kcd_forward, get_date_range
import numpy as np
import pandas as pd
from typing import Final, Union

class ICIS:
    '''
    ICIS (Insurance Credit Information System) Claim Information

    Abbreviations:
        uw:  Underwriting - Insurance policy evaluation process
        clm: Claim - Insurance claim submission
        hos: Hospitalization - Medical treatment usually requiring overnight stay (including same-day stay)
        sur: Surgery - Surgical procedures performed
        out: Outpatient - Medical treatment without overnight stay
        elp: Elapsed - Time passed since an event
        day: Day - Number of days
        cnt: Count - Number of occurrences
        mod: Modified - Adjusted or changed value
        
    Column Naming Convention:
        hos_day: Total days of hospitalization per case
        hos_cnt: Number of separate hospitalization events
        sur_cnt: Number of surgical procedures performed
        elp_day: Number of days elapsed since most recent occurrence
        hos_edate_mod: Modified hospital discharge date

    Instance Variables:
        filled: DataFrame with forward-filled KCD codes
        melted: Long-format DataFrame after melting KCD columns
        hospitalized: DataFrame containing hospitalization day calculations
        underwent: DataFrame containing surgery count calculations
        elapsed: DataFrame containing elapsed day calculations
        merged: Final DataFrame with all metrics combined
    
    Processing Pipeline:
        1. Data validation
            - validate_columns(): Verify required columns and their order
        
        2. Data cleansing
            - drop_duplicates(): Remove duplicate records
            - fill_kcd_forward(): Forward fill KCD codes
            - filter_by_clm_date(): Filter by claim date
        
        3. Data preparation
            - set_type(): Set medical care types
            - set_hos_edate_mod(): Modify hospital end dates
            - melt(): Convert to long format
            - set_sub_kcd(): Set sub KCD codes
            - merge_main_info(): Merge main diagnosis info
            - filter_sub_kcd(): Filter sub KCD codes
        
        4. Date calculations
            - set_date_range(): Set hospitalization date ranges
            - calc_hos_day(): Calculate hospitalization days
            - calc_sur_cnt(): Calculate surgery counts
            - calc_elp_day(): Calculate elapsed days
        
        5. Final merge
            - merge_calculated(): Combine all calculated metrics
    
    Required Columns:
        data DataFrame:
            - id: Patient identifier
            - kcd0-kcd4: Diagnosis codes
            - clm_date: Claim date
            - hos_sdate: Hospital admission date
            - hos_edate: Hospital discharge date
            - hos_day: Length of stay
            - hos_cnt: Hospitalization count
            - out_cnt: Outpatient visit count
            - sur_cnt: Surgery count
        
        main DataFrame:
            - kcd: Diagnosis code
            - kcd_main: Main category code
            - sub_chk: Sub-diagnosis check flag
    
    Note:
        This class processes and analyzes medical claim data
        from the Insurance Credit Information System. It handles
        various medical events including hospitalizations,
        surgeries, and outpatient visits within specified
        time periods from the underwriting date.
    '''
    UW_DATE: Final[pd.Timestamp] = pd.Timestamp.today() # Default underwriting date (today)
    FILTER_DAYS: Final[int] = 1825 # Default lookback period (5 years: 365 * 5 days)

    def __init__(self, claim: pd.DataFrame, main: pd.DataFrame, uw_date: Union[str, pd.Timestamp] = UW_DATE, filter_days: int = FILTER_DAYS):
        # Create a copy of input data to preserve original
        self.claim = claim.copy()
        self.main = main

         # Convert date columns to datetime format
        date_columns = ['clm_date', 'hos_sdate', 'hos_edate']
        for col in date_columns:
            if col in self.claim.columns:
                self.claim[col] = pd.to_datetime(self.claim[col], format='%Y%m%d')

        # Initialize instance variables
        self.filled = None
        self.melted = None
        self.hospitalized = None
        self.underwent = None
        self.elapsed = None
        self.merged = None
        self.uw_date = pd.to_datetime(uw_date) # string or datetime
        self.filter_days = filter_days

    def validate_columns(self) -> None:
        """
        Validates if all required columns exist in claim and main DataFrames.
        
        Required columns:
            claim: ClaimColumns enum values in specified order
            main: MainColumns enum values
        
        Raises:
            ValueError: If any required column is missing or if there are unexpected columns
            ValueError: If columns are not in the expected order
        """
        # Check claim DataFrame columns existence
        claim_required = [col.value for col in ClaimColumns]
        claim_current = list(self.claim.columns)
        
        missing_claim_cols = set(claim_required) - set(claim_current)
        unexpected_claim_cols = set(claim_current) - set(claim_required)
        
        if missing_claim_cols:
            raise ValueError(
                f"Missing required columns in claim DataFrame: {sorted(missing_claim_cols)}\n"
                f"Required columns: {claim_required}"
            )
        
        if unexpected_claim_cols:
            raise ValueError(
                f"Unexpected columns found in claim DataFrame: {sorted(unexpected_claim_cols)}\n"
                f"Expected columns: {claim_required}"
            )
        
        # Check claim DataFrame column order
        if claim_current != claim_required:
            raise ValueError(
                f"Columns in claim DataFrame are not in the expected order.\n"
                f"Current order: {claim_current}\n"
                f"Expected order: {claim_required}"
            )
        
        # Check main DataFrame columns
        main_required = [col.value for col in MainColumns]
        main_current = list(self.main.columns)
        
        missing_main_cols = set(main_required) - set(main_current)
        unexpected_main_cols = set(main_current) - set(main_required)
        
        if missing_main_cols:
            raise ValueError(
                f"Missing required columns in main DataFrame: {sorted(missing_main_cols)}\n"
                f"Required columns: {main_required}"
            )
        
        if unexpected_main_cols:
            raise ValueError(
                f"Unexpected columns found in main DataFrame: {sorted(unexpected_main_cols)}\n"
                f"Expected columns: {main_required}"
            )

    def drop_duplicates(self) -> pd.DataFrame:
        self.claim.drop_duplicates(subset=["id", "kcd0", "kcd1", "kcd2", "kcd3", "kcd4", "clm_date", "hos_sdate", "hos_edate", "hos_day", "hos_cnt", "sur_cnt", "out_cnt"], keep='first', inplace=True)
        self.claim.reset_index(drop=True, inplace=True)
        return self.claim

    def fill_kcd_forward(self) -> pd.DataFrame:
        self.filled = fill_kcd_forward(self.claim)
        return self.filled

    def filter_by_clm_date(self) -> pd.DataFrame:
        """
        Filters out claims that are older than self.filter_days from the underwriting date.

        Formula: 
            uw_date - claim_date <= filter_days

        Example:
            If filter_days = 1825 (5 years = 365 * 5) and uw_date is 2024-01-01:
            - A claim from 2019-01-01 will be kept (1825 days)
            - A claim from 2018-12-31 will be filtered out (1826 days)

        Note:
            Uses self.filter_days and self.uw_date set during initialization.
            Resets index after filtering.

        Returns:
            DataFrame: Filtered data containing only recent claims within the specified period
        """

        # Calculate date difference and filter
        mask = self.uw_date - self.filled['clm_date'] <= pd.Timedelta(days=self.filter_days)
        self.filled = self.filled[mask]
        self.filled.reset_index(drop=True, inplace=True)
        return self.filled

    def set_type(self) -> pd.DataFrame:
        """
        Set medical care type based on hospitalization, surgery, and outpatient treatment.
        
        Types:
            - 'hos': Hospitalization
            - 'sur': Surgery
            - 'out': Outpatient
        
        Returns:
            DataFrame: Data with medical care type assigned
        """
        self.filled['type'] = np.where((self.filled['hos_day'] >  0) & (self.filled['sur_cnt'] >  0), 'hos/sur', 
                              np.where((self.filled['hos_day'] >  0), 'hos', 
                              np.where((self.filled['sur_cnt'] >  0), 'sur', 
                              np.where((self.filled['hos_day'] == 0) & (self.filled['sur_cnt'] == 0), 'out', None))))
        return self.filled

    def set_hos_edate_mod(self) -> pd.DataFrame:
        """
        Modify hospital end date based on type and hos_day.
        Uses .loc to avoid SettingWithCopyWarning.
        
        Returns:
            DataFrame: Data with modified hospital end dates
        """
        if self.filled is None:
            raise ValueError("Must call set_type() before set_hos_edate_mod()")

        # Create a copy to ensure we're working with a view
        self.filled = self.filled.copy()
        
        # Add hos_edate_mod column initialized with hos_edate
        self.filled.loc[:, 'hos_edate_mod'] = self.filled['hos_edate']
        
        # Update hos_edate_mod for hospitalization cases
        hos_mask = self.filled['type'].str.contains('hos', na=False)
        self.filled.loc[hos_mask, 'hos_edate_mod'] = (
            self.filled.loc[hos_mask, 'hos_sdate'] + 
            pd.to_timedelta(self.filled.loc[hos_mask, 'hos_day'] - 1, unit='days')
        )
        
        return self.filled

    def melt(self) -> pd.DataFrame:
        """
        Melt KCD columns into long format and convert kcd_ord to numeric values.
        Example: 'kcd0' -> 0, 'kcd1' -> 1, etc.
        Excludes rows where kcd is NaN.
        
        Returns:
            DataFrame: Melted data with numeric kcd_ord values
        """
        if self.filled is None:
            raise ValueError("Must call fill_kcd_forward() before melt()")
        
        # Melt KCD columns
        self.melted = pd.melt(
            self.filled, 
            id_vars=ID_COLS, 
            value_vars=KCD_COLS, 
            var_name='kcd_ord', 
            value_name='kcd'
        )
        
        # Convert kcd_ord from 'kcd0' to 0
        self.melted['kcd_ord'] = self.melted['kcd_ord'].str.replace('kcd', '').astype(int)

        # Remove rows where kcd is NaN
        self.melted.dropna(subset=['kcd'], inplace=True)
        #self.melted.reset_index(drop=True, inplace=True)
        
        return self.melted

    def set_sub_kcd(self) -> pd.DataFrame:
        self.melted['sub_kcd'] = np.where(self.melted['kcd_ord'] == 0, 0, 1)
        return self.melted

    def merge_main_info(self) -> pd.DataFrame:
        """
        Merge kcd_main and sub_chk columns from main DataFrame to data DataFrame.
        Matching is done based on the 'kcd' column.
        
        Returns:
            DataFrame: data merged with main information
        """
        if self.melted is None:
            raise ValueError("Must call melt() before merge_main_info()")
        
        # Merge main information with melted data
        self.melted = self.melted.merge(
            self.main[['kcd', 'kcd_main', 'sub_chk']], 
            how='left',
            on='kcd'
        )
        
        return self.melted
    
    def filter_sub_kcd(self) -> pd.DataFrame:
        """
        Filter out rows where sub_kcd=1 and sub_chk=0.
        This removes secondary diagnosis codes that are not marked as significant.
        
        Returns:
            DataFrame: Filtered data excluding non-significant secondary diagnoses
        """
        if self.melted is None:
            raise ValueError("Must call melt() before filter_sub_kcd()")
            
        # Keep rows that are either:
        # 1. Main diagnosis (sub_kcd != 1) OR
        # 2. Significant secondary diagnosis (sub_chk == 1)
        mask = ~((self.melted['sub_kcd'] == 1) & (self.melted['sub_chk'] == 0))
        self.melted = self.melted[mask].reset_index(drop=True)
        
        return self.melted
    
    def set_date_range(self) -> pd.DataFrame:
        """
        Set date range for hospitalization records
        """
        if self.melted is None:
            raise ValueError("Must call melt() before set_date_range()")
        
        # 1. Initialize hos_vdate column with empty lists using numpy array
        arr = np.empty(len(self.melted), dtype=object)
        arr.fill([])
        self.melted['hos_vdate'] = arr

        # 2. Create mask for hospitalization cases
        mask = self.melted['type'].str.contains('hos', na=False)

        # 3. Calculate and assign date ranges for masked indices
        date_ranges = get_date_range(
            self.melted.loc[mask, 'hos_sdate'].values,
            self.melted.loc[mask, 'hos_edate_mod'].values
        )

        # 4. Assign date ranges using numpy array indexing
        self.melted['hos_vdate'].values[mask] = date_ranges
        self.melted.reset_index(drop=True, inplace=True)

        return self.melted

    def calc_hos_day(self) -> pd.DataFrame:
        """
        Combines hos_vdate lists by id and kcd_main, removes duplicates, and counts unique dates.
        Groups by main diagnosis code instead of individual KCD codes.
        
        Returns:
            DataFrame: Summarized data with unique hospitalization date counts per id, kcd_main group
        """
        # Filter for hospitalization cases and calculate unique dates
        self.hospitalized = (self.melted[self.melted['type'].str.contains('hos', na=False)]
            .groupby(['id', 'kcd_main'])
            .agg(
                hos_day = ('hos_vdate', lambda x: len(set([date for dates in x for date in dates if date])))
            )
            .reset_index()
        )

        return self.hospitalized

    def calc_sur_cnt(self) -> pd.DataFrame:
        """
        Calculate unique surgery dates by id and kcd_main.
        Groups by main diagnosis code instead of individual KCD codes.
        Only counts rows where sur_cnt > 0.
        
        Returns:
            DataFrame: Number of unique surgery dates per id, kcd_main group
        """
        # Filter for surgery cases and count unique dates by kcd_main
        self.underwent = (self.melted[self.melted['sur_cnt'] > 0]
            .groupby(['id', 'kcd_main'])
            .agg(
                sur_cnt = ('clm_date', 'nunique')
            )
            .reset_index()
        )
        
        return self.underwent

    def calc_elp_day(self) -> pd.DataFrame:
        """
        Calculate elapsed days for SI (hospitalization/surgery) and STD (all events)
        Groups by main diagnosis code instead of individual KCD codes.

        Formula:
            hos: uw_date - hos_edate_mod + 1
            sur: uw_date - clm_date + 1
            out: uw_date - clm_date + 1

        Note: 
            - Larger values indicate events that occurred further in the past
            - For hospitalization cases, uses hos_edate_mod
            - For other cases, uses clm_date

        Definitions:
            - elp_day_si: Elapsed days for hospitalization/surgery events only
            - elp_day_std: Elapsed days for all medical event

        Returns:
            DataFrame: Elapsed days by id and kcd_main, with columns:
                - id: Patient identifier
                - kcd_main: Main diagnosis code group
                - elp_day_si: Days elapsed since most recent hospitalization/surgery
                - elp_day_std: Days elapsed since most recent medical event
        """
        # Calculate elapsed days for each KCD code
        self.melted['elp_day'] = np.where(
            self.melted['type'].str.contains('hos', na=False),
            (self.uw_date - self.melted['hos_edate_mod'] + pd.Timedelta(days=1)).dt.days,
            (self.uw_date - self.melted['clm_date'] + pd.Timedelta(days=1)).dt.days
        )
        
        # Create mask for hos/sur events
        si_mask = self.melted['type'].str.contains('hos|sur', na=False)
        
        # Calculate elapsed days separately for SI and STD
        self.elapsed = (self.melted
            .groupby(['id', 'kcd_main'])
            .agg(
                elp_day_si = pd.NamedAgg(
                    column='elp_day',
                    aggfunc=lambda x: x[si_mask].min() if x[si_mask].any() else None
                ),
                elp_day_std = pd.NamedAgg(
                    column='elp_day',
                    aggfunc=lambda x: x.min()
                )
            )
            .reset_index()
        )
        
        return self.elapsed

    def merge_calculated(self) -> pd.DataFrame:
        """
        Merge hospitalization, surgery, and elapsed day information by id and kcd_main.
        
        Returns:
            DataFrame: Summarized data with columns:
                - id: Patient identifier
                - kcd_main: Main diagnosis code group
                - hos_day: Total days of hospitalization (0 if None)
                - sur_cnt: Number of surgeries (0 if None)
                - elp_day_si: Days elapsed since most recent hospital/surgery event
                - elp_day_std: Days elapsed since most recent medical event
        """
        # Start with hospitalization data
        self.merged = self.hospitalized.merge(
            self.underwent, 
            on=['id', 'kcd_main'], 
            how='outer'
        )
        
        # Add elapsed days information
        self.merged = self.merged.merge(
            self.elapsed,
            on=['id', 'kcd_main'],
            how='outer'
        )

        # Replace None with 0 for hos_day and sur_cnt
        self.merged['hos_day'] = self.merged['hos_day'].fillna(0)
        self.merged['sur_cnt'] = self.merged['sur_cnt'].fillna(0)
        
        return self.merged

    def process(self) -> pd.DataFrame:
        """
        Execute the complete ICIS data processing pipeline.
        
        Processing steps:
        1. Data cleaning
            - Remove duplicates
            - Forward fill KCD codes
            - Filter by claim date
        
        2. Data preparation
            - Set medical care types
            - Modify hospitalization end dates
            - Melt data to long format
            - Set and merge main KCD codes
            - Filter sub KCD codes
        
        3. Date calculations
            - Set date ranges
            - Calculate hospitalization days
            - Calculate surgery counts
            - Calculate elapsed days
        
        4. Final merge
            - Combine all calculated metrics
        
        Returns:
            DataFrame: Final processed data with all medical metrics
        """
        # 1. Data validation
        self.validate_columns()

        # 2. Data cleansing
        self.drop_duplicates()
        self.fill_kcd_forward()
        self.filter_by_clm_date()
        
        # 3. Data preparation
        self.set_type()
        self.set_hos_edate_mod()
        self.melt()
        self.set_sub_kcd()
        self.merge_main_info()
        self.filter_sub_kcd()
        
        # 4. Date calculations
        self.set_date_range()
        self.calc_hos_day()
        self.calc_sur_cnt()
        self.calc_elp_day()
        
        # 5. Final merge
        return self.merge_calculated()
