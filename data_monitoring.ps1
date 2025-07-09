# CMP Pad Quality Pipeline - Data Monitoring Script
# Fixed version with proper PowerShell string handling

Write-Host "Starting CMP Pad Quality Data Monitoring..." -ForegroundColor Green

# Quality metrics query with proper string handling
Write-Host "`n[Data] Hourly Quality Trends" -ForegroundColor Cyan

$qualityMetricsQuery = "SELECT COUNT(*) as total_records, COUNT(CASE WHEN created_at >= NOW() - INTERVAL '1 hour' THEN 1 END) as last_hour, COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as last_24_hours FROM cmp_data.pad_quality"

# Execute quality metrics query
try {
    $result = psql -h localhost -d cmp_warehouse -c $qualityMetricsQuery
    Write-Host $result -ForegroundColor White
} catch {
    Write-Host "Error executing quality metrics query: $($_.Exception.Message)" -ForegroundColor Red
}

# Quality trend analysis with proper syntax
$trendAnalysisQuery = "SELECT CASE WHEN COUNT(CASE WHEN is_faulty = true THEN 1 END) * 100.0 / COUNT(*) > 15 THEN 'High defect rate detected' ELSE 'Quality within acceptable range' END as quality_status, ROUND(COUNT(CASE WHEN is_faulty = true THEN 1 END) * 100.0 / COUNT(*), 2) as defect_rate_percent FROM cmp_data.pad_quality WHERE created_at >= NOW() - INTERVAL '1 hour'"

try {
    $trendResult = psql -h localhost -d cmp_warehouse -c $trendAnalysisQuery
    Write-Host $trendResult -ForegroundColor White
} catch {
    Write-Host "Error executing trend analysis query: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n[Data] Latest 5 Records" -ForegroundColor Cyan

# Latest records query
$latestRecordsQuery = "SELECT pad_id, quality_score, is_faulty, created_at FROM cmp_data.pad_quality ORDER BY created_at DESC LIMIT 5"

try {
    $latestResult = psql -h localhost -d cmp_warehouse -c $latestRecordsQuery
    Write-Host $latestResult -ForegroundColor White
} catch {
    Write-Host "Error executing latest records query: $($_.Exception.Message)" -ForegroundColor Red
}

# Model performance query
$modelPerformanceQuery = "SELECT model_version, ROUND(accuracy, 4) as accuracy_score, ROUND(precision_score, 4) as precision, ROUND(recall_score, 4) as recall FROM cmp_data.model_performance WHERE created_at >= NOW() - INTERVAL '24 hours' ORDER BY created_at DESC"

Write-Host "`n[Analytics] Prediction Analytics (Last 24 Hours)" -ForegroundColor Cyan

try {
    $modelResult = psql -h localhost -d cmp_warehouse -c $modelPerformanceQuery
    Write-Host $modelResult -ForegroundColor White
} catch {
    Write-Host "Error executing model performance query: $($_.Exception.Message)" -ForegroundColor Red
}

# Kafka connectivity check
Write-Host "`n[System] Kafka Connectivity Status" -ForegroundColor Cyan

try {
    # Check Kafka topics
    $kafkaTopics = kafka-topics --bootstrap-server localhost:9092 --list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Kafka connection successful" -ForegroundColor Green
        Write-Host "Available topics:" -ForegroundColor White
        Write-Host $kafkaTopics -ForegroundColor Gray
    } else {
        Write-Host "[ERROR] Kafka connection failed" -ForegroundColor Red
        Write-Host $kafkaTopics -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Error checking Kafka connectivity: $($_.Exception.Message)" -ForegroundColor Red
}

# Database connectivity check
Write-Host "`n[System] Database Connectivity Status" -ForegroundColor Cyan

try {
    $dbTest = psql -h localhost -d cmp_warehouse -c "SELECT 1 as connection_test;" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Database connection successful" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Database connection failed" -ForegroundColor Red
        Write-Host $dbTest -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Error checking database connectivity: $($_.Exception.Message)" -ForegroundColor Red
}

# Additional Kafka consumer lag check
Write-Host "`n[System] Kafka Consumer Lag Check" -ForegroundColor Cyan

try {
    $consumerGroups = kafka-consumer-groups --bootstrap-server localhost:9092 --list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Consumer groups found:" -ForegroundColor Green
        Write-Host $consumerGroups -ForegroundColor Gray
        
        # Check lag for each consumer group
        foreach ($group in $consumerGroups -split "`n") {
            if ($group.Trim() -ne "") {
                $lagInfo = kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group $group.Trim() 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Lag info for group $($group.Trim()):" -ForegroundColor Yellow
                    Write-Host $lagInfo -ForegroundColor Gray
                }
            }
        }
    } else {
        Write-Host "[WARNING] No consumer groups found or Kafka not accessible" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[ERROR] Error checking Kafka consumer lag: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n[COMPLETE] Monitoring script completed" -ForegroundColor Green