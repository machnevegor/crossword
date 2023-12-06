## Benchmark

To analyze the algorithm's performance statistically, 100 random input files
were generated (see [testdata](./testdata/) folder). The distribution of the
number of words in the input file is shown in the following figure:

| Word count | Number of files |
| ---------- | --------------- |
| 7          | 8               |
| 8          | 12              |
| 9          | 24              |
| 10         | 28              |
| 11         | 19              |
| 12         | 5               |
| 13         | 4               |

The tests were executed on [Ubuntu 22.04](https://releases.ubuntu.com/22.04/)
using [Python 3.10.12](https://www.python.org/downloads/release/python-31012/).
A total of 6000 runs were performed, i.e. 60 runs for each of the 100 test
inputs (see [testdata](./testdata/) folder). The results were saved to
[benchmarks.json](./benchmark/benchmarks.json) with the following model:

```python
from enum import Enum

from pydantic import BaseModel


class Status(str, Enum):
    """The status of the crossword generation."""

    OK = "ok"
    """The crossword was successfully generated."""
    DEGENERATED = "degenerated"
    """During the crossword generation, the population has degenerated."""


class Launch(BaseModel):
    """The instance of crossword generation launch."""

    time: int
    """The time of crossword generation."""
    status: Status
    """The status of crossword generation."""


class TimeStat(BaseModel):
    """The time stats for the crossword generation."""

    size: int
    """The number of representatives in the sample."""
    mean: int | None
    """The mean of the sample. None (null) if the sample is empty."""
    variance: int | None
    """The variance of the sample. None (null) if the sample is empty
    or consists of only one representative.
    """


class LaunchStat(BaseModel):
    """The launch stats for the crossword generation."""

    success: TimeStat
    """The time stats for the successful launches."""
    failure: TimeStat
    """The time stats for the failed launches."""
    total: TimeStat
    """The time stats for overall launches."""


class Benchmark(BaseModel):
    """Benchmark for crossword generation based on multiple runs of a
    specific test file.
    """

    filename: str
    """The path to the test input file."""

    words: int
    """The number of words in the input file."""
    genes: int
    """The total number of word intersections possible."""

    launches: list[Launch]
    """The list of launches."""

    stats: LaunchStat
    """Time stats based on the examined launches."""
```

**Note** that the following generation parameters were used for benchmarking:

| Constant                               | Description                                                    | Value           |
| -------------------------------------- | -------------------------------------------------------------- | --------------- |
| [WORD_PATTERN](./main.py#L18)          | Regular expression to assert a crossword word.                 | `^[a-z]{2,20}$` |
| [ROW_SIZE](./main.py#L21)              | Row size of the crossword grid.                                | `20`            |
| [COL_SIZE](./main.py#L23)              | Column size of the crossword grid.                             | `20`            |
| [MUTATION_ATTEMPTS](./main.py#L26)     | The number of attempts to mutate an individual.                | `10`            |
| [GENOME_EXTENSION_RATE](./main.py#L28) | The probability that a mutation will lead to genome expansion. | `0.5`           |
| [GENOME_SHRINKAGE_RATE](./main.py#L30) | The probability that a mutation will lead to genome shrinkage. | `0.15`          |
| [POPULATION_LIMIT](./main.py#L32)      | The maximum number of individuals in the population.           | `256`           |
