#!/usr/bin/env python3

import json
import random
import time
import argparse
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CMPPadDataGenerator:
    def __init__(self):
        self.batch_counter = 1
        self.pad_counter = 1
    
    def generate_pad_data(self):
        """Generate synthetic CMP pad data"""
        batch_id = f"BATCH_{self.batch_counter:06d}"
        pad_id = f"PAD_{self.pad_counter:08d}"
        
        # Generate realistic CMP parameters
        thickness = round(random.uniform(0.8, 1.2), 4)
        pressure = round(random.uniform(2.0, 8.0), 4)
        temperature = round(random.uniform(20.0, 35.0), 4)
        rotation_speed = round(random.uniform(50.0, 200.0), 4)
        
        # Simulate fault conditions (10% chance of fault)
        is_faulty = random.random() < 0.1
        
        # If faulty, skew the parameters
        if is_faulty:
            thickness += random.uniform(-0.3, 0.3)
            pressure += random.uniform(-2.0, 2.0)
            temperature += random.uniform(-5.0, 5.0)
            rotation_speed += random.uniform(-20.0, 20.0)
        
        data = {
            'batch_id': batch_id,
            'pad_id': pad_id,
            'thickness': thickness,
            'pressure': pressure,
            'temperature': temperature,
            'rotation_speed': rotation_speed,
            'is_faulty': is_faulty,
            'timestamp': datetime.now().isoformat()
        }
        
        self.pad_counter += 1
        if self.pad_counter % 100 == 0:
            self.batch_counter += 1
        
        return data

def create_kafka_producer():
    """Create and return Kafka producer"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=['kafka:29092'],  # Updated for Docker network
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retry_backoff_ms=100,
            request_timeout_ms=30000,
            retries=3
        )
        logger.info("Kafka producer created successfully")
        return producer
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise

def send_messages(producer, topic, num_messages):
    """Send messages to Kafka topic"""
    generator = CMPPadDataGenerator()
    
    for i in range(num_messages):
        try:
            data = generator.generate_pad_data()
            future = producer.send(topic, value=data)
            
            # Wait for message to be sent
            record_metadata = future.get(timeout=10)
            
            logger.info(f"Message {i+1}/{num_messages} sent successfully: "
                       f"Topic={record_metadata.topic}, "
                       f"Partition={record_metadata.partition}, "
                       f"Offset={record_metadata.offset}")
            
            time.sleep(0.1)
            
        except KafkaError as e:
            logger.error(f"Failed to send message {i+1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending message {i+1}: {e}")
    
    producer.flush()
    logger.info(f"All {num_messages} messages sent successfully")

def main():
    parser = argparse.ArgumentParser(description='CMP Pad Data Kafka Producer')
    parser.add_argument('--messages', type=int, default=10, 
                       help='Number of messages to send (default: 10)')
    parser.add_argument('--topic', type=str, default='cmp_pad_topic',
                       help='Kafka topic name (default: cmp_pad_topic)')
    
    args = parser.parse_args()
    
    try:
        producer = create_kafka_producer()
        send_messages(producer, args.topic, args.messages)
        producer.close()
        logger.info("Producer closed successfully")
    except Exception as e:
        logger.error(f"Producer failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())