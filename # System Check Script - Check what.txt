# System Check Script - Check what tools are available

Write-Host "=== System Tools Check ===" -ForegroundColor Yellow

# Check PostgreSQL
Write-Host "`n[PostgreSQL Tools]" -ForegroundColor Cyan
try {
    $psqlVersion = psql --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] psql found: $psqlVersion" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] psql not found in PATH" -ForegroundColor Red
    }
} catch {
    Write-Host "[MISSING] psql not found in PATH" -ForegroundColor Red
}

# Check for alternative PostgreSQL tools
$possiblePsqlPaths = @(
    "C:\Program Files\PostgreSQL\*\bin\psql.exe",
    "C:\Program Files (x86)\PostgreSQL\*\bin\psql.exe",
    "$env:LOCALAPPDATA\Programs\PostgreSQL\*\bin\psql.exe"
)

foreach ($path in $possiblePsqlPaths) {
    $found = Get-ChildItem -Path $path -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "[FOUND] PostgreSQL at: $($found.FullName)" -ForegroundColor Yellow
        Write-Host "Add to PATH: $($found.Directory.FullName)" -ForegroundColor Yellow
    }
}

# Check Kafka tools
Write-Host "`n[Kafka Tools]" -ForegroundColor Cyan
try {
    $kafkaVersion = kafka-topics --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] kafka-topics found: $kafkaVersion" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] kafka-topics not found in PATH" -ForegroundColor Red
    }
} catch {
    Write-Host "[MISSING] kafka-topics not found in PATH" -ForegroundColor Red
}

# Check for Kafka installation
$possibleKafkaPaths = @(
    "C:\kafka*\bin\windows\kafka-topics.bat",
    "C:\Program Files\kafka*\bin\windows\kafka-topics.bat",
    "$env:USERPROFILE\kafka*\bin\windows\kafka-topics.bat"
)

foreach ($path in $possibleKafkaPaths) {
    $found = Get-ChildItem -Path $path -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "[FOUND] Kafka at: $($found.FullName)" -ForegroundColor Yellow
        Write-Host "Add to PATH: $($found.Directory.FullName)" -ForegroundColor Yellow
    }
}

# Check Docker (alternative approach)
Write-Host "`n[Docker (Alternative)]" -ForegroundColor Cyan
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Docker found: $dockerVersion" -ForegroundColor Green
        Write-Host "You can use Docker for PostgreSQL and Kafka" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] Docker not found" -ForegroundColor Red
    }
} catch {
    Write-Host "[MISSING] Docker not found" -ForegroundColor Red
}

# Check Python (alternative for database connections)
Write-Host "`n[Python (Alternative)]" -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
        Write-Host "You can use Python with psycopg2 for PostgreSQL" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] Python not found" -ForegroundColor Red
    }
} catch {
    Write-Host "[MISSING] Python not found" -ForegroundColor Red
}

Write-Host "`n=== Installation Instructions ===" -ForegroundColor Yellow
Write-Host "1. PostgreSQL Client: https://www.postgresql.org/download/windows/" -ForegroundColor White
Write-Host "2. Kafka: https://kafka.apache.org/downloads" -ForegroundColor White
Write-Host "3. Or use Docker: docker run -it --rm postgres:15 psql --help" -ForegroundColor White