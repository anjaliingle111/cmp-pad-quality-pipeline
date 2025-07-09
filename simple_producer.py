import json
import random
import time
from datetime import datetime
from kafka import KafkaProducer

def create_sample_data():
    return {
        'batch_id': f'BATCH_{random.randint(1, 1000):06d}',
        'pad_id': f'PAD_{random.randint(1, 10000):08d}',
        'thickness': round(random.uniform(0.8, 1.2), 4),
        'pressure': round(random.uniform(2.0, 8.0), 4),
        'temperature': round(random.uniform(20.0, 35.0), 4),
        'rotation_speed': round(random.uniform(50.0, 200.0), 4),
        'is_faulty': random.random() < 0.1,
        'timestamp': datetime.now().isoformat()
    }

def main():
    # Create producer
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Create topic
    print("Creating topic...")
    
    # Send 10 messages
    for i in range(10):
        data = create_sample_data()
        producer.send('cmp_pad_topic', value=data)
        print(f"Sent message {i+1}: {data['pad_id']}")
        time.sleep(0.5)
    
    producer.flush()
    producer.close()
    print("Done!")

if __name__ == '__main__':
    main()