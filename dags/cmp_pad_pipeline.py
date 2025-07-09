from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import json
import random
import psycopg2
from kafka import KafkaProducer, KafkaConsumer

default_args = {
    'owner': 'data-engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'cmp_pad_pipeline',
    default_args=default_args,
    description='CMP Pad Quality Pipeline',
    schedule_interval=timedelta(minutes=15),
    catchup=False,
    tags=['cmp', 'kafka', 'manufacturing']
)

def generate_and_send_data():
    """Generate CMP pad data and send to Kafka"""
    print("Starting data generation...")
    
    producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Generate 20 sample records
    for i in range(20):
        data = {
            'batch_id': f'BATCH_{random.randint(1, 1000):06d}',
            'pad_id': f'PAD_{random.randint(1, 10000):08d}',
            'thickness': round(random.uniform(0.8, 1.2), 4),
            'pressure': round(random.uniform(2.0, 8.0), 4),
            'temperature': round(random.uniform(20.0, 35.0), 4),
            'rotation_speed': round(random.uniform(50.0, 200.0), 4),
            'is_faulty': random.random() < 0.1,
            'timestamp': datetime.now().isoformat()
        }
        
        producer.send('cmp_pad_topic', value=data)
        print(f"Sent record {i+1}: {data['pad_id']}")
    
    producer.flush()
    producer.close()
    print("Data generation completed!")

def process_kafka_data():
    """Process data from Kafka and store in database"""
    print("Starting data processing...")
    
    # Create database connection
    conn = psycopg2.connect(
        host='postgres',
        database='cmp_warehouse',
        user='airflow',
        password='airflow',
        port='5432'
    )
    cursor = conn.cursor()
    
    # Ensure table exists
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
    
    # Create Kafka consumer
    consumer = KafkaConsumer(
        'cmp_pad_topic',
        bootstrap_servers=['kafka:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=30000,
        auto_offset_reset='latest',
        group_id='airflow_processor'
    )
    
    processed_count = 0
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
        processed_count += 1
        print(f"Processed record {processed_count}: {data['pad_id']}")
    
    cursor.close()
    conn.close()
    consumer.close()
    print(f"Processing completed! Processed {processed_count} records")

def generate_quality_report():
    """Generate quality analysis report"""
    print("Generating quality report...")
    
    conn = psycopg2.connect(
        host='postgres',
        database='cmp_warehouse',
        user='airflow',
        password='airflow',
        port='5432'
    )
    cursor = conn.cursor()
    
    # Generate quality statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN is_faulty = true THEN 1 END) as faulty_records,
            ROUND(AVG(thickness), 4) as avg_thickness,
            ROUND(AVG(pressure), 4) as avg_pressure,
            ROUND(AVG(temperature), 4) as avg_temperature,
            ROUND(AVG(rotation_speed), 4) as avg_rotation_speed
        FROM cmp_data.pad_quality
        WHERE created_at >= NOW() - INTERVAL '1 hour';
    """)
    
    result = cursor.fetchone()
    
    if result:
        total, faulty, avg_thickness, avg_pressure, avg_temp, avg_rotation = result
        fault_rate = (faulty / total * 100) if total > 0 else 0
        
        report = f"""
        CMP Pad Quality Report (Last Hour)
        ===================================
        Total Records: {total}
        Faulty Records: {faulty}
        Fault Rate: {fault_rate:.2f}%
        
        Average Measurements:
        - Thickness: {avg_thickness} mm
        - Pressure: {avg_pressure} psi
        - Temperature: {avg_temp} °C
        - Rotation Speed: {avg_rotation} RPM
        """
        
        print(report)
        
        # Store report
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cmp_data.quality_reports (
                id SERIAL PRIMARY KEY,
                report_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cursor.execute("""
            INSERT INTO cmp_data.quality_reports (report_text)
            VALUES (%s)
        """, (report,))
        
        conn.commit()
    
    cursor.close()
    conn.close()
    print("Quality report generated!")

# Define tasks
generate_data_task = PythonOperator(
    task_id='generate_cmp_data',
    python_callable=generate_and_send_data,
    dag=dag
)

process_data_task = PythonOperator(
    task_id='process_kafka_data',
    python_callable=process_kafka_data,
    dag=dag
)

quality_report_task = PythonOperator(
    task_id='generate_quality_report',
    python_callable=generate_quality_report,
    dag=dag
)

# Set dependencies
generate_data_task >> process_data_task >> quality_report_task