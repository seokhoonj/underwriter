# Underwriter Package

**Author**: Seokhoon Joo

## Overview
The Underwriter package provides tools for processing and analyzing insurance claim data, with a focus on the Insurance Credit Information System (ICIS). It helps Korean insurance underwriters evaluate medical histories and assess risks based on hospitalization, surgery, and outpatient records.

## Installation

```bash
pip install git+https://github.com/seokhoonj/underwriter.git
```

## Usage

### Required Data
The package requires two main data inputs:

1. **ICIS Claim Data**: Insurance claim history containing:
   - Unique identifier for each insured person or policy
   - Primary diagnosis code
   - Secondary diagnosis codes
   - Date when the claim was filed
   - Hospital admission and discharge dates
   - Number of hospitalization days and counts
   - Number of outpatient visits
   - Number of surgeries

2. **Main Disease Classification Data**: Reference data for disease categorization and filterting:
   - KCD (Korean Classification of Diseases) codes
   - Main disease categories
   - Filtering criteria

### Basic Example
```python
from underwriter.icis import ICIS

# Initialize ICIS with your data
icis = ICIS(claim=ICIS_Claim_Data, main=Main_Disease_Classification_Data)

# Process the data
result = icis.process()
```

For detailed examples and step-by-step guide, see [examples/notebooks/01.ICIS-Claim-Data-Processing.ipynb](examples/notebooks/01.ICIS-Claim-Data-Processing.ipynb)
