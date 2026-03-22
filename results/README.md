# Results

This directory stores benchmark run outputs.

## File Format

Each run produces a JSON file with the following structure:

```json
{
  "timestamp": "2024-01-15T12:00:00+00:00",
  "config": { ... },
  "models": {
    "GPT-4o": [
      {
        "category": "jailbreak",
        "total_tests": 16,
        "passed": 15,
        "failed": 1,
        "pass_rate": 0.9375,
        "duration_seconds": 45.2
      }
    ]
  },
  "overall_scores": {
    "GPT-4o": 91.5
  }
}
```

## Generated Reports

- `benchmark_results.json` - Raw result data
- `report.md` - Detailed Markdown report
- `leaderboard.md` - Model ranking table
