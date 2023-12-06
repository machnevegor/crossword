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
