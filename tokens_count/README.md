# tokens_count

This folder contains scripts and files for tracking and reporting token usage for AI models in the project.

## Files

- `usage_tracker.py`: Python script to track and log token usage for various AI operations.
- `total_tokens.txt`: Cumulative count of tokens used across all operations.
- `usage_report.txt`: Detailed report of token usage, including breakdowns by operation or time period.

Use these files to monitor and optimize your project's AI resource consumption.

IMPORTANT NOTES ⚠️

- This report of tokens is only an ESTIMATION because some usage in the
  very first days was not logged.
- The total includes both embedding tokens and 4o tokens.
- Token logging inside the main project code was later removed entirely,
  since the API key will not be used anymore after project submission.
