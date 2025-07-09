# CMP Pad Quality Pipeline - Hybrid Monitoring Script
# Uses local PostgreSQL and Docker for Kafka

Write-Host "Starting CMP Pad Quality Data Monitoring (Hybrid)..." -ForegroundColor Green

# Configuration
$psqlPath = "C:\Program Files\PostgreSQL\15\bin\psql.exe"
$dbHost = "127.0.0.1"
$dbName = "cmp_warehouse"
$dbUser = "postgres"  # Use postgres user

# Quality metrics query
Write-Host "`n[Data] Hourly Quality Trends" -ForegroundColor Cyan

$qualityMetricsQuery = "SELECT COUNT(*) as total_records, COUNT(CASE WHEN created_at >= NOW() - INTERVAL '1 hour' THEN 1 END) as last_hour, COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as last_24_hours FROM cmp_data.pad_quality"

# Execute quality metrics query using full path
try {
    $result = & $psqlPath -h $dbHost -d $dbName -U $dbUser -c $qualityMetricsQuery 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $result -ForegroundColor White
    } else {
        Write-Host "[ERROR] Database query failed. Check credentials." -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to execute query: $($_.Exception.Message)" -ForegroundColor Red
}

# Quality trend analysis
$trendAnalysisQuery = "SELECT CASE WHEN COUNT(CASE WHEN is_faulty = true THEN 1 END) * 100.0 / COUNT(*) > 15 THEN 'High defect rate detected' ELSE 'Quality within acceptable range' END as quality_status, ROUND(COUNT(CASE WHEN is_faulty = true THEN 1 END) * 100.0 / COUNT(*), 2) as defect_rate_percent FROM cmp_data.pad_quality WHERE created_at >= NOW() - INTERVAL '1 hour'"

try {
    $trendResult = & $psqlPath -h $dbHost -d $dbName -U $dbUser -c $trendAnalysisQuery 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $trendResult -ForegroundColor White
    } else {
        Write-Host "[ERROR] Trend analysis query failed" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to execute trend analysis: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n[Data] Latest 5 Records" -ForegroundColor Cyan

# Latest records query
$latestRecordsQuery = "SELECT pad_id, quality_score, is_faulty, created_at FROM cmp_data.pad_quality ORDER BY created_at DESC LIMIT 5"

try {
    $latestResult = & $psqlPath -h $dbHost -d $dbName -U $dbUser -c $latestRecordsQuery 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $latestResult -ForegroundColor White
    } else {
        Write-Host "[ERROR] Latest records query failed" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to execute latest records query: $($_.Exception.Message)" -ForegroundColor Red
}

# Model performance query
$modelPerformanceQuery = "SELECT model_version, ROUND(accuracy, 4) as accuracy_score, ROUND(precision_score, 4) as precision, ROUND(recall_score, 4) as recall FROM cmp_data.model_performance WHERE created_at >= NOW() - INTERVAL '24 hours' ORDER BY created_at DESC"

Write-Host "`n[Analytics] Prediction Analytics (Last 24 Hours)" -ForegroundColor Cyan

try {
    $modelResult = & $psqlPath -h $dbHost -d $dbName -U $dbUser -c $modelPerformanceQuery 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $modelResult -ForegroundColor White
    } else {
        Write-Host "[ERROR] Model performance query failed" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to execute model performance query: $($_.Exception.Message)" -ForegroundColor Red
}

# Kafka connectivity check using Docker
Write-Host "`n[System] Kafka Connectivity Status (Docker)" -ForegroundColor Cyan

try {
    # Check if Kafka is running by trying to connect
    $kafkaTest = docker run --rm -it --network host confluentinc/cp-kafka:latest timeout 5 kafka-topics --bootstrap-server localhost:9092 --list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Kafka connection successful" -ForegroundColor Green
        Write-Host "Available topics:" -ForegroundColor White
        Write-Host $kafkaTest -ForegroundColor Gray
    } else {
        Write-Host "[ERROR] Kafka connection failed" -ForegroundColor Red
        Write-Host "Make sure Kafka is running on localhost:9092" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[ERROR] Error checking Kafka connectivity: $($_.Exception.Message)" -ForegroundColor Red
}

# Database connectivity check
Write-Host "`n[System] Database Connectivity Status" -ForegroundColor Cyan

try {
    $dbTest = & $psqlPath -h $dbHost -d $dbName -U $dbUser -c "SELECT 1 as connection_test;" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Database connection successful" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Database connection failed" -ForegroundColor Red
        Write-Host $dbTest -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Error checking database connectivity: $($_.Exception.Message)" -ForegroundColor Red
}

# Kafka consumer groups check using Docker
Write-Host "`n[System] Kafka Consumer Groups (Docker)" -ForegroundColor Cyan

try {
    $consumerGroups = docker run --rm -it --network host confluentinc/cp-kafka:latest timeout 5 kafka-consumer-groups --bootstrap-server localhost:9092 --list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Consumer groups found:" -ForegroundColor Green
        Write-Host $consumerGroups -ForegroundColor Gray
    } else {
        Write-Host "[WARNING] No consumer groups found or Kafka not accessible" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[ERROR] Error checking Kafka consumer groups: $($_.Exception.Message)" -ForegroundColor Red
}

# Network connectivity tests
Write-Host "`n[System] Network Connectivity Tests" -ForegroundColor Cyan

# Test PostgreSQL port
try {
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $tcpClient.Connect("localhost", 5432)
    $tcpClient.Close()
    Write-Host "[OK] PostgreSQL port 5432 is reachable" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] PostgreSQL port 5432 is not reachable" -ForegroundColor Red
}

# Test Kafka port
try {
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $tcpClient.Connect("localhost", 9092)
    $tcpClient.Close()
    Write-Host "[OK] Kafka port 9092 is reachable" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Kafka port 9092 is not reachable" -ForegroundColor Red
}

Write-Host "`n=== Troubleshooting Tips ===" -ForegroundColor Yellow
Write-Host "1. If database queries fail, set PGPASSWORD environment variable" -ForegroundColor White
Write-Host "2. If Kafka tests fail, ensure Kafka is running: docker-compose up kafka" -ForegroundColor White
Write-Host "3. Check your database and Kafka configurations" -ForegroundColor White

Write-Host "`n[COMPLETE] Hybrid monitoring completed" -ForegroundColor Green
