"""
Data Loader — Loads, caches, and filters transaction and customer data.

Provides a singleton pattern for in-memory caching and convenient
filter methods for the agent tools to use.
"""

import os
import pandas as pd
from typing import Optional
from pathlib import Path


import logging
logger = logging.getLogger("aml_sentinel")

class DataLoader:
    """
    Singleton data loader with in-memory caching.
    Loads transaction and customer CSVs and provides filter utilities.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.transactions: Optional[pd.DataFrame] = None
        self.customers: Optional[pd.DataFrame] = None
        self._initialized = True
    
    def load(self, transactions_path: str, customers_path: Optional[str] = None):
        """Load data from CSV files."""
        if os.path.exists(transactions_path):
            # Load max 50,000 rows to prevent OOM on massive Kaggle datasets
            df = pd.read_csv(transactions_path, nrows=50000)
            
            # Detect Kaggle IBM AML Dataset schema
            if 'Account' in df.columns and 'Account.1' in df.columns:
                logger.info("Detected IBM AML Dataset schema. Mapping columns to internal schema...")
                df = df.rename(columns={
                    'Account': 'customer_id',
                    'Account.1': 'counterparty_id',
                    'Timestamp': 'timestamp',
                    'Amount Paid': 'amount',
                    'Payment Format': 'transaction_type',
                    'Is Laundering': 'ground_truth_flag'
                })
                if 'Payment Currency' in df.columns:
                    df['currency'] = df['Payment Currency']
                
                # IBM datasets often use numeric hashes for accounts
                df['customer_id'] = df['customer_id'].astype(str)
                df['counterparty_id'] = df['counterparty_id'].astype(str)
            
            self.transactions = df
            
            # Parse timestamps
            if 'timestamp' in self.transactions.columns:
                self.transactions['timestamp'] = pd.to_datetime(
                    self.transactions['timestamp'], errors='coerce'
                )
            logger.info(f"Loaded transactions: {len(self.transactions):,} rows (capped at 50K for memory safety)")
        else:
            raise FileNotFoundError(f"Transaction file not found: {transactions_path}")
        
        if customers_path and os.path.exists(customers_path):
            self.customers = pd.read_csv(customers_path)
            if 'account_open_date' in self.customers.columns:
                self.customers['account_open_date'] = pd.to_datetime(
                    self.customers['account_open_date'], errors='coerce'
                )
            print(f"  📄 Loaded customers: {len(self.customers):,} rows")
        else:
            self.customers = pd.DataFrame()
            print("  ⚠️ No customer data file found")
    
    def get_dataset_info(self) -> dict:
        """Get dataset metadata."""
        if self.transactions is None:
            return {"error": "No data loaded"}
        
        info = {
            "name": "AML Sentinel Synthetic Dataset",
            "rows": len(self.transactions),
            "columns": len(self.transactions.columns),
            "column_names": list(self.transactions.columns),
            "unique_customers": int(self.transactions['customer_id'].nunique())
                if 'customer_id' in self.transactions.columns else 0,
            "total_transactions": len(self.transactions),
        }
        
        if 'timestamp' in self.transactions.columns:
            info["date_range"] = (
                f"{self.transactions['timestamp'].min().strftime('%Y-%m-%d')} to "
                f"{self.transactions['timestamp'].max().strftime('%Y-%m-%d')}"
            )
        
        if 'ground_truth_flag' in self.transactions.columns:
            suspicious = self.transactions[self.transactions['ground_truth_flag'] == True]
            info["suspicious_count"] = len(suspicious)
            if 'ground_truth_pattern' in suspicious.columns:
                info["pattern_distribution"] = suspicious['ground_truth_pattern'].value_counts().to_dict()
        
        if self.customers is not None and len(self.customers) > 0:
            info["total_customers"] = len(self.customers)
        
        return info
    
    def filter_transactions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        customer_ids: Optional[list] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        transaction_types: Optional[list] = None,
        countries: Optional[list] = None,
    ) -> pd.DataFrame:
        """Filter transactions based on criteria."""
        if self.transactions is None:
            return pd.DataFrame()
        
        df = self.transactions.copy()
        
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        if customer_ids:
            df = df[df['customer_id'].isin(customer_ids)]
        if min_amount is not None:
            df = df[df['amount'] >= min_amount]
        if max_amount is not None:
            df = df[df['amount'] <= max_amount]
        if transaction_types:
            df = df[df['transaction_type'].isin(transaction_types)]
        if countries and 'counterparty_country' in df.columns:
            df = df[df['counterparty_country'].isin(countries)]
        
        return df
    
    def get_customer(self, customer_id: str) -> Optional[dict]:
        """Get details for a specific customer."""
        if self.customers is None or len(self.customers) == 0:
            return None
        cust = self.customers[self.customers['customer_id'] == customer_id]
        if cust.empty:
            return None
        return cust.iloc[0].to_dict()
