from enum import Enum

from pydantic import BaseModel


class Status(str, Enum):
    """The status of the crossword generation."""

    OK = "ok"
    """The crossword was successfully generated."""
    DEGENERATED = "degenerated"
    """During the crossword generation, the population has degenerated."""


class SuccessLaunch(BaseModel):
    """The successful execution of crossword generation."""

    status: Status = Status.OK
    """The status of crossword generation."""
    time: int
    """The time of crossword generation."""
    fitness: int
    """The fitness of the generated crossword."""


class FailureLaunch(BaseModel):
    """The failed execution of crossword generation."""

    status: Status = Status.DEGENERATED
    """The status of crossword generation."""


Launch = SuccessLaunch | FailureLaunch
"""The instance of crossword generation launch."""


class SampleRange(BaseModel):
    """The range of the sample values."""

    min: int
    """The minimum value in the sample."""
    max: int
    """The maximum value in the sample."""


class SampleStat(BaseModel):
    """The statistical characteristics of the sample."""

    size: int
    """The number of representatives in the sample."""
    mean: int
    """The mean of the sample."""
    standard_deviation: int
    """The standard deviation of the sample."""
    range: SampleRange
    """The range of the sample."""


class LaunchStat(BaseModel):
    """The launch stats for crossword generation."""

    time: SampleStat
    """The time stats for successful launches."""
    fitness: SampleStat
    """The fitness stats for the launches."""
    setbacks: int
    """The amount of failures during the launches."""


class Benchmark(BaseModel):
    """Benchmark for crossword generation based on multiple runs of a
    given sample input file.
    """

    filename: str
    """The path to the input file."""

    words: int
    """The number of words in the input file."""
    genes: int
    """The total number of word intersections possible."""

    launches: list[Launch]
    """The launch list with execution information."""

    stats: LaunchStat
    """Stats based on the examined launches."""
