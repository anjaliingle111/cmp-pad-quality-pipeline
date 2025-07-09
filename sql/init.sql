-- Create database schema for CMP pad data
CREATE SCHEMA IF NOT EXISTS cmp_data;

-- Create table for CMP pad quality data
CREATE TABLE IF NOT EXISTS cmp_data.pad_quality (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(50) NOT NULL,
    pad_id VARCHAR(50) NOT NULL,
    thickness DECIMAL(10,4),
    pressure DECIMAL(10,4),
    temperature DECIMAL(10,4),
    rotation_speed DECIMAL(10,4),
    is_faulty BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_pad_quality_batch_id ON cmp_data.pad_quality(batch_id);
CREATE INDEX IF NOT EXISTS idx_pad_quality_created_at ON cmp_data.pad_quality(created_at);
CREATE INDEX IF NOT EXISTS idx_pad_quality_is_faulty ON cmp_data.pad_quality(is_faulty);