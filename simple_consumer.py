import json
import psycopg2
from kafka import KafkaConsumer

def create_database_table():
    conn = psycopg2.connect(
        host='localhost',
        database='cmp_warehouse',
        user='airflow',
        password='airflow',
        port='5432'
    )
    cursor = conn.cursor()
    
    # Create schema and table
    cursor.execute("""
        CREATE SCHEMA IF NOT EXISTS cmp_data;
        
        CREATE TABLE IF NOT EXISTS cmp_data.pad_quality (
            id SERIAL PRIMARY KEY,
            batch_id VARCHAR(50) NOT NULL,
            pad_id VARCHAR(50) NOT NULL,
            thickness DECIMAL(10,4),
            pressure DECIMAL(10,4),
            temperature DECIMAL(10,4),
            rotation_speed DECIMAL(10,4),
            is_faulty BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Database table created!")

def consume_messages():
    # Create consumer
    consumer = KafkaConsumer(
        'cmp_pad_topic',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=10000,
        auto_offset_reset='earliest'
    )
    
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        database='cmp_warehouse',
        user='airflow',
        password='airflow',
        port='5432'
    )
    cursor = conn.cursor()
    
    messages_processed = 0
    for message in consumer:
        data = message.value
        
        # Insert into database
        cursor.execute("""
            INSERT INTO cmp_data.pad_quality 
            (batch_id, pad_id, thickness, pressure, temperature, rotation_speed, is_faulty)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            data['batch_id'],
            data['pad_id'],
            data['thickness'],
            data['pressure'],
            data['temperature'],
            data['rotation_speed'],
            data['is_faulty']
        ))
        
        conn.commit()
        messages_processed += 1
        print(f"Processed message {messages_processed}: {data['pad_id']}")
    
    cursor.close()
    conn.close()
    consumer.close()
    print(f"Finished processing {messages_processed} messages")

if __name__ == '__main__':
    create_database_table()
    consume_messages()