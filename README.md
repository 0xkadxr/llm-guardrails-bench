![CI](https://github.com/kadirou12333/llm-guardrails-bench/actions/workflows/ci.yml/badge.svg)

# LLM Guardrails Bench

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A standardized benchmark suite for evaluating LLM safety guardrails and defense mechanisms. Compare models across jailbreak resistance, content filtering, instruction hierarchy adherence, data leakage prevention, and refusal accuracy.

---

## Leaderboard (Example Results)

| Rank | Model | Jailbreak | Content Filter | Instruction Hierarchy | Data Leakage | Refusal Accuracy | Overall |
|:----:|:------|:---------:|:--------------:|:---------------------:|:------------:|:----------------:|:-------:|
| 1 | GPT-4o | 94% | 96% | 88% | 92% | 90% | **92.0** |
| 2 | Claude 3.5 Sonnet | 92% | 98% | 91% | 90% | 87% | **91.6** |
| 3 | Llama 3 70B | 78% | 85% | 72% | 80% | 82% | **79.4** |

*Placeholder results for illustration. Run the benchmark to generate real scores.*

---

## Features

- **Multi-model support** -- Evaluate OpenAI, Anthropic, and local HuggingFace models through a unified interface
- **5 benchmark categories** -- Jailbreak resistance, content filtering, instruction hierarchy, data leakage, and refusal accuracy
- **Extensible architecture** -- Add custom benchmarks, models, and datasets with minimal boilerplate
- **Rich CLI output** -- Progress bars, summary tables, and colored output via Rich
- **Automated reporting** -- Generate Markdown reports and leaderboard tables
- **Statistical analysis** -- Confidence intervals, category breakdowns, and composite scoring
- **Parallel execution** -- Run benchmarks concurrently for faster evaluation
- **Reproducible** -- Configurable seeds and deterministic evaluation pipelines

## Quick Start

### Installation

```bash
git clone https://github.com/kadirou12333/llm-guardrails-bench.git
cd llm-guardrails-bench
pip install -e ".[dev]"
```

### Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Run the Benchmark

```python
from guardrails_bench import BenchmarkRunner
from guardrails_bench.config import ModelConfig, ModelProvider, BenchmarkConfig

models = [
    ModelConfig(name="GPT-4o", provider=ModelProvider.OPENAI, model_id="gpt-4o"),
    ModelConfig(name="Claude 3.5 Sonnet", provider=ModelProvider.ANTHROPIC, model_id="claude-3-5-sonnet-20241022"),
]

runner = BenchmarkRunner(models=models)
results = runner.run()
results.save("results/benchmark_results.json")
```

Or use the example script:

```bash
python examples/run_benchmark.py
```

## Benchmark Categories

### 1. Jailbreak Resistance

Tests the model's ability to resist manipulation attempts across five attack vectors:

| Technique | Description |
|-----------|-------------|
| **Role-playing** | Asks the model to adopt an unrestricted persona (e.g., "DAN" prompts) |
| **Encoding** | Uses Base64, ROT13, or reversed text to obfuscate harmful requests |
| **Hypothetical** | Frames harmful requests as fictional, academic, or creative exercises |
| **Multi-turn** | Gradually escalates from benign to harmful across conversational turns |
| **Authority** | Claims admin/developer override permissions to bypass safety |

### 2. Content Filtering

Measures how effectively the model blocks harmful content generation across categories (violence, illegal activity, PII generation, malware, fraud) while avoiding false positives on benign queries.

### 3. Instruction Hierarchy

Tests whether the model properly prioritizes system-level instructions over conflicting user messages. Includes adherence tests (does the model follow its system prompt?) and extraction tests (can a user extract the system prompt content?).

### 4. Data Leakage

Evaluates resistance to data exfiltration attempts including PII generation, credential fabrication, training data extraction, and system information disclosure.

### 5. Refusal Accuracy

Measures the precision of the model's safety guardrails by testing both true positive rate (correctly refusing harmful requests) and false positive rate (incorrectly blocking benign requests).

## Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Attack Success Rate** | Percentage of attacks that bypassed guardrails | 0% (safest) -- 100% |
| **False Positive Rate** | Percentage of benign requests incorrectly blocked | 0% (best) -- 100% |
| **Robustness Score** | Composite safety score: 80% blocking effectiveness + 20% usability | 0 -- 100 |
| **Category Breakdown** | Per-technique pass rates for granular analysis | 0% -- 100% |

## Adding Custom Benchmarks

Create a new benchmark by implementing the evaluate interface:

```python
from guardrails_bench.config import BenchmarkCategory, BenchmarkConfig
from guardrails_bench.runner import SingleResult

class MyCustomBenchmark:
    CATEGORY = BenchmarkCategory.JAILBREAK  # or register a new category

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def evaluate(self, model) -> list[SingleResult]:
        results = []
        # Your evaluation logic here
        response = model.generate("Your test prompt")
        results.append(SingleResult(
            prompt_id="custom_01",
            category="custom",
            subcategory="my_test",
            passed=True,  # your pass/fail logic
            model_response=response,
            latency_ms=0.0,
        ))
        return results
```

## Adding Custom Models

Implement the `BaseModel` interface:

```python
from guardrails_bench.models.base import BaseModel
from guardrails_bench.config import ModelConfig, ModelRegistry

class MyModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Initialize your model client

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        # Call your model API and return the response text
        return "model response"

ModelRegistry.register("my_provider", MyModel)
```

## Project Structure

```
llm-guardrails-bench/
├── guardrails_bench/
│   ├── runner.py              # Main benchmark orchestrator
│   ├── config.py              # Configuration and model registry
│   ├── benchmarks/            # Benchmark implementations
│   ├── models/                # Model backend wrappers
│   ├── metrics/               # Scoring and statistical analysis
│   ├── datasets/data/         # Test prompt datasets (JSON)
│   └── reporting/             # Report and leaderboard generation
├── examples/                  # Usage examples
├── tests/                     # Unit tests
└── results/                   # Benchmark output directory
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{llm_guardrails_bench,
  title={LLM Guardrails Bench: A Standardized Benchmark for LLM Safety Guardrails},
  author={Benmeriem, Abdelkader},
  year={2024},
  url={https://github.com/kadirou12333/llm-guardrails-bench},
  license={MIT}
}
```

## Ethical Disclaimer

This benchmark suite is designed for **defensive security research** to help developers evaluate and improve the safety of LLM deployments. The test prompts included in the datasets are intentionally adversarial and are provided solely for the purpose of measuring guardrail effectiveness.

**Important:**
- Do not use the jailbreak prompts or harmful request datasets for any purpose other than safety evaluation.
- Results from this benchmark should be used to **improve** safety mechanisms, not to circumvent them.
- Always follow responsible disclosure practices if you discover vulnerabilities in production LLM systems.
- This tool does not guarantee comprehensive safety coverage. Use it as one component of a broader safety evaluation strategy.

## License

MIT License. See [LICENSE](LICENSE) for details.
