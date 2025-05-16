import json
from datetime import datetime
import pandas as pd
import os
import shutil
from pathlib import Path
import glob
import uuid
import time
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Local directory configuration
RAW_DIR = './raw'
MLFLOW_DATA_DIR = '../MLFlow/data'

def clear_previous_data():
    """Clear all previous data from raw directory"""
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Cleared and recreated {RAW_DIR} directory")

def write_batch_to_json(records):
    """Write a batch of records to JSON files"""
    for record in records:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_string = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{random_string}.json"
        filepath = os.path.join(RAW_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(record, f, indent=2)

def generate_faker_batch(batch_size=100):
    """Generate a batch of click events and food orders"""
    records = []
    
    # Click event data
    click_types = ['button', 'link', 'image']
    pages = ['/home', '/products', '/cart', '/checkout', '/profile', '/search', '/categories', '/deals']
    
    # Food order data
    restaurants = ['Pizza Place', 'Burger Joint', 'Sushi Bar', 'Taco Stand', 'Noodle House']
    items = ['pizza', 'burger', 'sushi', 'taco', 'noodles', 'salad', 'drink', 'dessert']
    
    for _ in range(batch_size):
        # Generate click event
        click_event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': fake.uuid4(),
            'page_url': random.choice(pages),
            'element_id': f'element_{fake.random_int(min=1000, max=9999)}',
            'session_id': fake.uuid4(),
            'click_type': random.choice(click_types)
        }
        records.append(click_event)
        
        # Generate food order with some random issues
        issues = random.sample(['missing', 'invalid', 'malformed', 'incomplete'], k=random.randint(1, 3))
        food_order = {
            'order_time': datetime.now().isoformat(),
            'customer_id': None if 'missing' in issues else (
                'invalid_id' if 'invalid' in issues else fake.uuid4()
            ),
            'restaurant': None if 'missing' in issues else random.choice(restaurants),
            'items': "invalid_items_format" if 'malformed' in issues else random.sample(items, random.randint(1, 4)),
            'total_amount': None if 'missing' in issues else (
                -1 if 'invalid' in issues else round(random.uniform(10.0, 100.0), 2)
            ),
            'delivery_address': None if 'missing' in issues else (
                'Incomplete Address' if 'incomplete' in issues else fake.address()
            )
        }
        records.append(food_order)
    
    return records

def process_parquet_files():
    """Process Parquet files in batches and convert to JSON"""
    parquet_files = glob.glob(os.path.join(MLFLOW_DATA_DIR, "*.parquet")) 
    
    for parquet_file in parquet_files:
        try:
            print(f"\nProcessing {os.path.basename(parquet_file)}...")
            start_time = time.time()
            
            # Read Parquet file
            df = pd.read_parquet(parquet_file, engine='pyarrow')
            
            # Process in batches of 1000 records
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                records = batch.to_dict('records')
                write_batch_to_json(records)
            
            end_time = time.time()
            file_records = len(df)
            total_records += file_records
            print(f"Processed {file_records:,} records in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing {parquet_file}: {str(e)}")
    
    print(f"\nTotal Parquet records processed: {total_records:,}")

def stream_faker_data(num_batches=10, batch_size=100):
    """Stream Faker-generated data in batches"""
    print("\nStreaming Faker-generated data...")
    total_records = 0
    start_time = time.time()
    
    for i in range(num_batches):
        try:
            # Generate and write a batch of records
            records = generate_faker_batch(batch_size)
            write_batch_to_json(records)
            
            total_records += len(records)
            print(f"Batch {i+1}/{num_batches}: Generated {len(records)} records")
            
            # Small delay between batches
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in batch {i+1}: {str(e)}")
    
    end_time = time.time()
    print(f"\nTotal Faker records generated: {total_records:,}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def main():
    # Clear previous data
    clear_previous_data()
    
    # Process Parquet files
    print("Starting data processing...")
    process_parquet_files()
    
    # Stream Faker data
    stream_faker_data(num_batches=10, batch_size=100)  # Generate 1000 records of each type
    
    print("\nData processing complete!")
    print(f"Data has been written to: {RAW_DIR}")

if __name__ == "__main__":
    main() 