# Health Tracker Sync Lambda Function

## Monitoring and Logging

The function emits the following metrics through CloudWatch:

### Operation Metrics
- `OperationAttempt` - Total number of operations attempted
- `GetOperation` - Number of GET operations
- `SyncOperation` - Number of SYNC operations
- `ExecutionTime` - Duration of each operation in milliseconds

### Error Metrics
- `MissingUserIdError` - Missing userId in request
- `MissingDataError` - Missing data in SYNC operation
- `InvalidOperationError` - Invalid operation type
- `UnhandledError` - Unhandled exceptions

### CloudWatch Logs
All metrics are logged with the format:
```
METRIC|{metricName}|{value}|Count
```

Example log entries:
```
METRIC|OperationAttempt|1|Count
METRIC|SyncOperation|1|Count
METRIC|ExecutionTime|123|Count
```

## Error Handling
1. All errors are logged with full context
2. Metrics are emitted for each error type
3. Client receives appropriate error messages
4. CORS headers are included in all responses

## Monitoring Dashboard
To create a monitoring dashboard:

1. Open CloudWatch
2. Create widgets for:
   - Operation success rate
   - Error rate by type
   - Average execution time
   - Request volume

## Alerts
Configure alerts for:
1. High error rate (>10% in 5 minutes)
2. Long execution times (>1s average)
3. Sustained error conditions
4. Unusual request patterns
