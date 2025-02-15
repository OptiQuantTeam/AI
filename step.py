import requests
import json

STEP_FUNCTION_ARN = "arn:aws:states:ap-northeast-2:123456789012:stateMachine:YourStateMachineName"

payload = {
    "stateMachineArn": STEP_FUNCTION_ARN,
    "input": json.dumps({"action": "BUY", "amount": 0.01, "symbol": "BTC/USD"})
}

headers = {
    "Content-Type": "application/x-amz-json-1.0",
    "X-Amz-Target": "AWSStepFunctions.StartExecution",
    "Authorization": "YOUR_AWS_AUTH_HEADER"
}

response = requests.post("https://states.ap-northeast-2.amazonaws.com", data=json.dumps(payload), headers=headers)
print(response.text)
