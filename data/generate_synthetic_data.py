import argparse
import os
import random
import csv
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import numpy as np
from faker import Faker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
HIGH_RISK_COUNTRIES = ['KY', 'PA', 'VG', 'BS', 'BZ', 'MM', 'KP', 'IR', 'SY']
NORMAL_COUNTRIES = ['US', 'UK', 'CA', 'DE', 'FR'] + ['US'] * 10  # Weight US heavily
ACCOUNT_TYPES = ['savings', 'checking', 'business']
KYC_STATUS = ['verified', 'pending', 'expired']
TRANSACTION_TYPES = ['deposit', 'withdrawal', 'transfer', 'wire_transfer']
CHANNELS = ['branch', 'online', 'atm', 'mobile']
MERCHANT_CATEGORIES = ['retail', 'dining', 'travel', 'utilities', 'financial_services', 'other']


def generate_customers(num_customers: int, fake: Faker) -> List[Dict[str, Any]]:
    logger.info(f"Generating {num_customers} customers...")
    customers = []
    for i in range(1, num_customers + 1):
        customer_id = f"C-{i:04d}"
        country = random.choices([random.choice(NORMAL_COUNTRIES), random.choice(HIGH_RISK_COUNTRIES)], weights=[0.95, 0.05])[0]
        
        customers.append({
            'customer_id': customer_id,
            'name': fake.name(),
            'age': random.randint(18, 80),
            'occupation': fake.job(),
            'account_type': random.choice(ACCOUNT_TYPES),
            'account_open_date': fake.date_between(start_date='-10y', end_date='today').isoformat(),
            'country': country,
            'risk_category': random.choices(['low', 'medium', 'high'], weights=[0.7, 0.2, 0.1])[0],
            'pep_flag': random.random() < 0.05,
            'kyc_status': random.choices(KYC_STATUS, weights=[0.9, 0.08, 0.02])[0],
        })
    return customers


def generate_normal_transactions(customers: List[Dict[str, Any]], num_transactions: int, fake: Faker) -> List[Dict[str, Any]]:
    logger.info("Generating normal transactions...")
    transactions = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    customer_ids = [c['customer_id'] for c in customers]
    
    for i in range(1, num_transactions + 1):
        txn_id = f"TXN-{fake.uuid4().split('-')[0].upper()}{i:04d}"
        cust_id = random.choice(customer_ids)
        
        # 80% small retail ($5-$500), 20% larger ($500-$5000)
        if random.random() < 0.8:
            amount = round(random.uniform(5, 500), 2)
        else:
            amount = round(random.uniform(500, 5000), 2)
            
        txn_type = random.choices(TRANSACTION_TYPES, weights=[0.3, 0.3, 0.3, 0.1])[0]
        is_cash = txn_type in ['deposit', 'withdrawal'] and random.random() < 0.5
        
        txn_date = fake.date_time_between(start_date=start_date, end_date=end_date)
        
        transactions.append({
            'transaction_id': txn_id,
            'customer_id': cust_id,
            'timestamp': txn_date.isoformat(),
            'amount': amount,
            'transaction_type': txn_type,
            'channel': random.choice(CHANNELS),
            'counterparty_id': random.choice(customer_ids) if txn_type in ['transfer', 'wire_transfer'] else '',
            'counterparty_country': random.choice(NORMAL_COUNTRIES) if txn_type in ['transfer', 'wire_transfer'] else '',
            'currency': 'USD',
            'account_balance': round(random.uniform(100, 50000), 2),
            'is_cash': is_cash,
            'merchant_category': random.choice(MERCHANT_CATEGORIES),
            'ground_truth_flag': False,
            'ground_truth_pattern': 'none'
        })
    
    return transactions


def inject_patterns(transactions: List[Dict[str, Any]], customers: List[Dict[str, Any]], fake: Faker) -> List[Dict[str, Any]]:
    logger.info("Injecting suspicious patterns...")
    
    customer_ids = [c['customer_id'] for c in customers]
    end_date = datetime.now()
    
    def generate_txn_id():
        return f"TXN-{fake.uuid4().split('-')[0].upper()}{random.randint(10000, 99999)}"
        
    pattern_txns = []
    
    # 1. STRUCTURING (~40 customers)
    structuring_custs = random.sample(customer_ids, 40)
    for cust in structuring_custs:
        base_date = fake.date_time_between(start_date='-300d', end_date='now')
        num_txns = random.randint(3, 7)
        for i in range(num_txns):
            txn_date = base_date + timedelta(days=random.uniform(0, 5))
            pattern_txns.append({
                'transaction_id': generate_txn_id(),
                'customer_id': cust,
                'timestamp': txn_date.isoformat(),
                'amount': round(random.uniform(8000, 9999), 2),
                'transaction_type': 'deposit',
                'channel': 'branch',
                'counterparty_id': '',
                'counterparty_country': '',
                'currency': 'USD',
                'account_balance': round(random.uniform(10000, 50000), 2),
                'is_cash': True,
                'merchant_category': 'financial_services',
                'ground_truth_flag': True,
                'ground_truth_pattern': 'STRUCTURING'
            })
            
    # 2. RAPID_MOVEMENT (~30 customers)
    rapid_custs = random.sample(list(set(customer_ids) - set(structuring_custs)), 30)
    for cust in rapid_custs:
        base_date = fake.date_time_between(start_date='-300d', end_date='now')
        amount = round(random.uniform(20000, 100000), 2)
        
        # Large deposit
        pattern_txns.append({
            'transaction_id': generate_txn_id(),
            'customer_id': cust,
            'timestamp': base_date.isoformat(),
            'amount': amount,
            'transaction_type': 'deposit',
            'channel': 'online',
            'counterparty_id': '',
            'counterparty_country': '',
            'currency': 'USD',
            'account_balance': round(random.uniform(100000, 200000), 2),
            'is_cash': False,
            'merchant_category': 'financial_services',
            'ground_truth_flag': True,
            'ground_truth_pattern': 'RAPID_MOVEMENT'
        })
        
        # Wire out within 24h
        wire_date = base_date + timedelta(hours=random.uniform(1, 23))
        pattern_txns.append({
            'transaction_id': generate_txn_id(),
            'customer_id': cust,
            'timestamp': wire_date.isoformat(),
            'amount': amount * random.uniform(0.9, 0.99), # Move almost all of it
            'transaction_type': 'wire_transfer',
            'channel': 'online',
            'counterparty_id': random.choice(customer_ids),
            'counterparty_country': random.choice(NORMAL_COUNTRIES),
            'currency': 'USD',
            'account_balance': round(random.uniform(1000, 5000), 2),
            'is_cash': False,
            'merchant_category': 'financial_services',
            'ground_truth_flag': True,
            'ground_truth_pattern': 'RAPID_MOVEMENT'
        })

    # 3. VELOCITY_SPIKE (~25 customers)
    velocity_custs = random.sample(list(set(customer_ids) - set(structuring_custs) - set(rapid_custs)), 25)
    for cust in velocity_custs:
        base_date = fake.date_time_between(start_date='-300d', end_date='now')
        num_txns = random.randint(15, 30) # 10x normal frequency in a week
        for _ in range(num_txns):
            txn_date = base_date + timedelta(days=random.uniform(0, 7))
            pattern_txns.append({
                'transaction_id': generate_txn_id(),
                'customer_id': cust,
                'timestamp': txn_date.isoformat(),
                'amount': round(random.uniform(100, 2000), 2),
                'transaction_type': 'transfer',
                'channel': 'mobile',
                'counterparty_id': random.choice(customer_ids),
                'counterparty_country': '',
                'currency': 'USD',
                'account_balance': round(random.uniform(5000, 20000), 2),
                'is_cash': False,
                'merchant_category': 'other',
                'ground_truth_flag': True,
                'ground_truth_pattern': 'VELOCITY_SPIKE'
            })

    # 4. DORMANT_ACTIVATION (~20 customers)
    dormant_custs = random.sample(list(set(customer_ids) - set(structuring_custs) - set(rapid_custs) - set(velocity_custs)), 20)
    for cust in dormant_custs:
        # Simulate activation after a long gap
        activation_date = fake.date_time_between(start_date='-100d', end_date='now')
        amount = round(random.uniform(10000, 50000), 2)
        pattern_txns.append({
            'transaction_id': generate_txn_id(),
            'customer_id': cust,
            'timestamp': activation_date.isoformat(),
            'amount': amount,
            'transaction_type': 'wire_transfer',
            'channel': 'online',
            'counterparty_id': random.choice(customer_ids),
            'counterparty_country': random.choice(NORMAL_COUNTRIES),
            'currency': 'USD',
            'account_balance': amount,
            'is_cash': False,
            'merchant_category': 'financial_services',
            'ground_truth_flag': True,
            'ground_truth_pattern': 'DORMANT_ACTIVATION'
        })

    # 5. GEOGRAPHIC_RISK (~25 customers)
    geo_custs = random.sample(list(set(customer_ids) - set(structuring_custs) - set(rapid_custs) - set(velocity_custs) - set(dormant_custs)), 25)
    for cust in geo_custs:
        txn_date = fake.date_time_between(start_date='-365d', end_date='now')
        pattern_txns.append({
            'transaction_id': generate_txn_id(),
            'customer_id': cust,
            'timestamp': txn_date.isoformat(),
            'amount': round(random.uniform(5000, 30000), 2),
            'transaction_type': 'wire_transfer',
            'channel': 'online',
            'counterparty_id': f"EXT-{fake.uuid4().split('-')[0]}",
            'counterparty_country': random.choice(HIGH_RISK_COUNTRIES),
            'currency': 'USD',
            'account_balance': round(random.uniform(10000, 50000), 2),
            'is_cash': False,
            'merchant_category': 'financial_services',
            'ground_truth_flag': True,
            'ground_truth_pattern': 'GEOGRAPHIC_RISK'
        })
        
    # 6. ROUND_TRIP (~15 customers)
    round_trip_groups = []
    available = list(set(customer_ids) - set(structuring_custs) - set(rapid_custs) - set(velocity_custs) - set(dormant_custs) - set(geo_custs))
    for _ in range(15):
        if len(available) >= 3:
            group = random.sample(available, 3)
            round_trip_groups.append(group)
            for g in group:
                available.remove(g)
                
    for group in round_trip_groups:
        a, b, c = group
        base_date = fake.date_time_between(start_date='-300d', end_date='now')
        amount = round(random.uniform(15000, 40000), 2)
        
        # A to B
        pattern_txns.append({
            'transaction_id': generate_txn_id(),
            'customer_id': a,
            'timestamp': base_date.isoformat(),
            'amount': amount,
            'transaction_type': 'transfer',
            'channel': 'online',
            'counterparty_id': b,
            'counterparty_country': '',
            'currency': 'USD',
            'account_balance': round(random.uniform(5000, 20000), 2),
            'is_cash': False,
            'merchant_category': 'financial_services',
            'ground_truth_flag': True,
            'ground_truth_pattern': 'ROUND_TRIP'
        })
        
        # B to C
        date2 = base_date + timedelta(hours=random.uniform(1, 12))
        pattern_txns.append({
            'transaction_id': generate_txn_id(),
            'customer_id': b,
            'timestamp': date2.isoformat(),
            'amount': amount * random.uniform(0.95, 0.99),
            'transaction_type': 'transfer',
            'channel': 'online',
            'counterparty_id': c,
            'counterparty_country': '',
            'currency': 'USD',
            'account_balance': round(random.uniform(5000, 20000), 2),
            'is_cash': False,
            'merchant_category': 'financial_services',
            'ground_truth_flag': True,
            'ground_truth_pattern': 'ROUND_TRIP'
        })
        
        # C to A
        date3 = date2 + timedelta(hours=random.uniform(1, 12))
        pattern_txns.append({
            'transaction_id': generate_txn_id(),
            'customer_id': c,
            'timestamp': date3.isoformat(),
            'amount': amount * random.uniform(0.90, 0.94),
            'transaction_type': 'transfer',
            'channel': 'online',
            'counterparty_id': a,
            'counterparty_country': '',
            'currency': 'USD',
            'account_balance': round(random.uniform(5000, 20000), 2),
            'is_cash': False,
            'merchant_category': 'financial_services',
            'ground_truth_flag': True,
            'ground_truth_pattern': 'ROUND_TRIP'
        })

    transactions.extend(pattern_txns)
    # Sort by timestamp
    transactions.sort(key=lambda x: x['timestamp'])
    return transactions

def verify_dataset(transactions: List[Dict[str, Any]], customers: List[Dict[str, Any]]):
    print("\n--- Dataset Verification ---")
    print(f"Total Customers: {len(customers)}")
    print(f"Total Transactions: {len(transactions)}")
    
    suspicious = [t for t in transactions if t['ground_truth_flag']]
    print(f"Suspicious Transactions: {len(suspicious)} ({len(suspicious)/len(transactions)*100:.2f}%)")
    
    patterns = {}
    for t in suspicious:
        pat = t['ground_truth_pattern']
        patterns[pat] = patterns.get(pat, 0) + 1
        
    print("\nPattern Breakdown:")
    for pat, count in patterns.items():
        print(f"  - {pat}: {count}")

def save_csv(data: List[Dict[str, Any]], filepath: str):
    if not data:
        return
    keys = data[0].keys()
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AML dataset")
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for CSV files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verify', action='store_true', help='Print verification stats')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)
    fake = Faker()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    customers = generate_customers(5000, fake)
    transactions = generate_normal_transactions(customers, 50000, fake)
    transactions = inject_patterns(transactions, customers, fake)
    
    if args.verify:
        verify_dataset(transactions, customers)
        
    cust_path = os.path.join(args.output_dir, 'sample_customers.csv')
    txn_path = os.path.join(args.output_dir, 'sample_transactions.csv')
    
    logger.info(f"Saving customers to {cust_path}")
    save_csv(customers, cust_path)
    
    logger.info(f"Saving transactions to {txn_path}")
    save_csv(transactions, txn_path)
    
    logger.info("Done.")

if __name__ == '__main__':
    main()
