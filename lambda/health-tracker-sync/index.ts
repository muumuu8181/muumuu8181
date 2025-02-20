import { APIGatewayEvent } from 'aws-lambda';
import { DynamoDBClient, QueryCommand, PutItemCommand } from '@aws-sdk/client-dynamodb';
import { CloudWatchLogsClient, PutLogEventsCommand } from '@aws-sdk/client-cloudwatch-logs';

const cloudwatch = new CloudWatchLogsClient({});
const LOG_GROUP = '/aws/lambda/health-tracker-sync';
const LOG_STREAM = 'monitoring';

const dynamodb = new DynamoDBClient({});
const TABLE_NAME = 'health-tracker-table';

async function logMetric(metricName: string, value: number = 1) {
  const timestamp = Date.now();
  const message = `METRIC|${metricName}|${value}|Count`;
  
  try {
    const command = new PutLogEventsCommand({
      logGroupName: LOG_GROUP,
      logStreamName: LOG_STREAM,
      logEvents: [{
        timestamp,
        message
      }]
    });
    
    await cloudwatch.send(command);
  } catch (error) {
    console.error('Failed to log metric:', error);
  }
}

interface HealthData {
  logs: {
    item: {
      name: string;
      amount: number;
    };
    type: 'food' | 'drink';
    time: string;
    date: string;
  }[];
  customFoodItems: string[];
  customDrinkItems: string[];
}

interface VersionedData extends HealthData {
  version: number;
  lastModified: number;
}

async function getUserData(userId: string): Promise<VersionedData | null> {
  const command = new QueryCommand({
    TableName: TABLE_NAME,
    KeyConditionExpression: 'userId = :userId',
    ExpressionAttributeValues: {
      ':userId': { S: userId }
    },
    Limit: 1,
    ScanIndexForward: false
  });

  const result = await dynamodb.send(command);
  if (!result.Items?.[0]) return null;

  const item = result.Items[0];
  return {
    ...item.data.M as unknown as HealthData,
    version: Number(item.version.N),
    lastModified: Number(item.lastModified.N)
  };
}

async function syncUserData(userId: string, data: VersionedData): Promise<VersionedData> {
  const current = await getUserData(userId);
  if (current && current.version >= data.version) {
    throw new Error('Conflict: Server has newer version');
  }

  const command = new PutItemCommand({
    TableName: TABLE_NAME,
    Item: {
      userId: { S: userId },
      version: { N: data.version.toString() },
      lastModified: { N: Date.now().toString() },
      data: { 
        M: {
          logs: { L: data.logs.map(log => ({ M: {
            item: { M: {
              name: { S: log.item.name },
              amount: { N: log.item.amount.toString() }
            }},
            type: { S: log.type },
            time: { S: log.time },
            date: { S: log.date }
          }}))},
          customFoodItems: { L: data.customFoodItems.map(item => ({ S: item })) },
          customDrinkItems: { L: data.customDrinkItems.map(item => ({ S: item })) }
        }
      }
    }
  });

  await dynamodb.send(command);
  return data;
}

export const handler = async (event: APIGatewayEvent) => {
  const startTime = Date.now();
  
  try {
    console.log('Event:', JSON.stringify(event));
    await logMetric('RequestReceived');
    
    const { userId, operation, data } = JSON.parse(event.body || '{}');

    if (!userId) {
      console.error('Missing userId');
      await logMetric('MissingUserIdError');
      return {
        statusCode: 400,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type'
        },
        body: JSON.stringify({ error: 'userId is required' })
      };
    }

    let result;
    console.log(`Processing ${operation} operation for user ${userId}`);
    await logMetric('OperationAttempt');
    
    switch (operation) {
      case 'GET':
        result = await getUserData(userId);
        await logMetric('GetOperation');
        break;
      case 'SYNC':
        if (!data) {
          console.error('Missing data for SYNC operation');
          await logMetric('MissingDataError');
          return {
            statusCode: 400,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*',
              'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
              'Access-Control-Allow-Headers': 'Content-Type'
            },
            body: JSON.stringify({ error: 'data is required for SYNC operation' })
          };
        }
        try {
          result = await syncUserData(userId, data);
          await logMetric('SyncOperation');
        } catch (error) {
          if (error instanceof Error && error.message === 'Conflict: Server has newer version') {
            await logMetric('VersionConflict');
            throw error;
          }
          throw error;
        }
        break;
      default:
        console.error(`Invalid operation: ${operation}`);
        return {
          statusCode: 400,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
          },
          body: JSON.stringify({ error: 'Invalid operation' })
        };
    }

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
      },
      body: JSON.stringify(result)
    };
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : 'Unknown error');
    await logMetric('Error');
    
    if (error instanceof Error && error.message === 'Conflict: Server has newer version') {
      return {
        statusCode: 409,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type'
        },
        body: JSON.stringify({ 
          error: 'Version conflict',
          message: 'Server has newer version'
        })
      };
    }
    
    return {
      statusCode: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
      },
      body: JSON.stringify({ error: 'Internal server error' })
    };
  } finally {
    const duration = Date.now() - startTime;
    await logMetric('ExecutionTime', duration);
  }
};
