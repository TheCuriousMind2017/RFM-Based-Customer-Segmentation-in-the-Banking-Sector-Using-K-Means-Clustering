"""Canonical schema for the Kaggle 'Bank Customer Segmentation' dataset.

Source: https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation
~1.05M transactions, ~800K+ customers, 9 raw columns. Indian bank, 2016.
Column names (including the parenthesised unit) are taken verbatim from the
source CSV so loading does not silently rename anything.
"""
from __future__ import annotations

# --- Raw column names (verbatim from the source CSV) ---
TRANSACTION_ID = "TransactionID"
CUSTOMER_ID = "CustomerID"
CUSTOMER_DOB = "CustomerDOB"
CUST_GENDER = "CustGender"
CUST_LOCATION = "CustLocation"
CUST_ACCOUNT_BALANCE = "CustAccountBalance"
TRANSACTION_DATE = "TransactionDate"
TRANSACTION_TIME = "TransactionTime"          # integer HHMMSS
TRANSACTION_AMOUNT = "TransactionAmount (INR)"

EXPECTED_COLUMNS = [
    TRANSACTION_ID,
    CUSTOMER_ID,
    CUSTOMER_DOB,
    CUST_GENDER,
    CUST_LOCATION,
    CUST_ACCOUNT_BALANCE,
    TRANSACTION_DATE,
    TRANSACTION_TIME,
    TRANSACTION_AMOUNT,
]

# Core columns required to compute RFM. Rows missing any of these are unusable.
RFM_REQUIRED = [CUSTOMER_ID, TRANSACTION_DATE, TRANSACTION_AMOUNT]

# --- Engineered / derived column names (stage 2 onward) ---
TRANSACTION_HOUR = "TransactionHour"
AGE = "Age"
DOB_VALID = "dob_valid"

# --- RFM modeling-table column names (stage 4 output) ---
RECENCY = "Recency"
FREQUENCY = "Frequency"
MONETARY = "Monetary"
MONETARY_MEAN = "MonetaryMean"
TENURE = "Tenure"
LAST_BALANCE = "LastAccountBalance"
DOMINANT_HOUR = "DominantHour"
