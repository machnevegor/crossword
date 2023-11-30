# Author: Egor Machnev
# Group: DSAI-03
# Email: e.machnev@innopolis.university

"""Assignment 2. Crossword Generation."""
from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from random import random, choice, sample
from re import match
from typing import Iterable

# --- CONSTANTS --- #

WORD_PATTERN = r"^[a-z]{2,20}$"
"""Regular expression to assert a crossword word."""

MUTATION_ATTEMPTS = 10
"""The number of attempts to mutate an individual."""
GENOME_EXTENSION_RATE = 0.5
"""The probability that a mutation will lead to genome expansion."""
GENOME_SHRINKAGE_RATE = 0.01
"""The probability that a mutation will lead to genome shrinkage."""

ROW_SIZE = 20
"""Row size of the crossword grid."""
COL_SIZE = 20
"""Column size of the crossword grid."""

POPULATION_LIMIT = 256
"""Population size limit after one evolutionary step."""

# --- DATA TYPES --- #


@dataclass(order=True, frozen=True)
class Char:
    """The letter in the crossword word."""

    word: str
    """The word the char belongs to."""
    index: int
    """The index of the char in the word."""

    @property
    def value(self) -> str:
        """The value of the char."""
        return self.word[self.index]

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        assert 0 <= self.index < len(self.word), "Invalid char index"


@dataclass(frozen=True)
class Gene:
    """The gene of the crossword genome."""

    a: Char
    """The A-component of the gene."""
    b: Char
    """The B-component of the gene."""

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        assert self.a < self.b, "Invalid order of gene components"

    def __iter__(self) -> Iterable[Char]:
        """Iterate over the gene components."""
        yield self.a
        yield self.b

    @classmethod
    def safe_init(cls, a: Char, b: Char) -> Gene:
        """Initialize the gene with the components in the proper order.

        Args:
            a (Char): The A-component of the gene.
            b (Char): The B-component of the gene.

        Returns:
            Gene: The gene with the components in the proper order.
        """
        return cls(a=a, b=b) if a < b else cls(a=b, b=a)


@dataclass(order=True, frozen=True)
class Loc:
    """The location on the crossword grid."""

    row: int
    """The row index on the crossword grid."""
    col: int
    """The column index on the crossword grid."""

    @property
    def T(self) -> Loc:
        """The transposed location."""
        return Loc(row=self.col, col=self.row)

    @classmethod
    def transform(cls, loc: Loc, origin: Loc) -> Loc:
        """Transform coordinate to new origin.

        Args:
            loc (Loc): The coordinate to transform.
            origin (Loc): The new origin.

        Returns:
            Loc: The transformed coordinate.
        """
        return cls(row=loc.row - origin.row, col=loc.col - origin.col)


@dataclass
class Cell:
    """The cell of the crossword grid."""

    hor: Char | None = field(default=None)
    """The horizontal char of the cell."""
    ver: Char | None = field(default=None)
    """The vertical char of the cell."""

    def __bool__(self) -> bool:
        return self.hor is not None or self.ver is not None


class Ori(Enum):
    """The direction the word goes in on the crossword grid."""

    HOR = 0
    """The word goes horizontally (left-to-right)."""
    VER = 1
    """The word goes vertically (top-to-bottom)."""

    @property
    def drow(self) -> int:
        """The row shift of the word."""
        return 1 if self == Ori.VER else 0

    @property
    def dcol(self) -> int:
        """The column shift of the word."""
        return 1 if self == Ori.HOR else 0

    def __invert__(self) -> Ori:
        """Get the opposite orientation."""
        return Ori.VER if self == Ori.HOR else Ori.HOR


@dataclass(frozen=True)
class WordHead:
    """The start of a word and its direction."""

    word: str
    """The word the head belongs to."""
    loc: Loc
    """The head's location on the crossword grid."""
    ori: Ori
    """The head's direction on the crossword grid."""


# --- I/O --- #


@dataclass
class Context:
    """The context of crossword generation."""

    words: list[str]
    """The list of words to generate the crossword from."""
    genes: set[Gene] = field(default_factory=set)
    """The set of all possible genes."""

    word_genes: defaultdict[str, set[Gene]] = field(
        default_factory=lambda: defaultdict(set)
    )
    """The set of genes corresponding to each word."""

    def __post_init__(self) -> None:
        """Post-initialization checks and preprocessing."""
        self._assert_words()
        self._comb_genes()

    @classmethod
    def from_file(cls, filename: str) -> Context:
        """Initialize the context from the input file.

        Args:
            filename (str): The path to the input file.

        Returns:
            Context: The context of crossword generation.
        """
        with open(filename, "r", encoding="utf-8") as file:
            lines = filter(None, map(str.strip, file))

            return cls(words=list(lines))

    def _assert_words(self) -> None:
        """Assert that all words are valid."""
        for word in self.words:
            assert match(WORD_PATTERN, word), "Invalid word"

    def _cross_words(self, a_word: str, b_word: str) -> None:
        """Find all possible intersections (genes) between the two
        words. Register the genes in the context.

        Args:
            a_word (str): The first word.
            b_word (str): The second word.
        """
        for i, a_char in enumerate(a_word):
            for j, b_char in enumerate(b_word):
                if a_char == b_char:
                    gene = Gene.safe_init(
                        a=Char(word=a_word, index=i), b=Char(word=b_word, index=j)
                    )

                    self.genes.add(gene)

                    self.word_genes[a_word].add(gene)
                    self.word_genes[b_word].add(gene)

    def _comb_genes(self) -> None:
        """Combine all possible genes from the words."""
        for i, a_word in enumerate(self.words):
            for j in range(i + 1, len(self.words)):
                self._cross_words(a_word, self.words[j])


def output_individual(context: Context, filename: str, individual: Individual) -> None:
    """Output the individual to the output file.

    Args:
        context (Context): The context of crossword generation.
        filename (str): The path to the output file.
        individual (Individual): The individual to output.
    """
    min_loc, max_loc = individual.boundaries

    # Check if the grid needs to be transposed.
    row_size = max_loc.row - min_loc.row + 1
    col_size = max_loc.col - min_loc.col + 1

    transposed = (
        (row_size >= ROW_SIZE or col_size >= COL_SIZE)
        and row_size < COL_SIZE
        and col_size < ROW_SIZE
    )

    with open(filename, "w", encoding="utf-8") as file:
        # Write word-by-word.
        for head in map(lambda word: individual.word_head[word], context.words):
            absolute_loc = Loc.transform(head.loc, min_loc)
            adjusted_loc = absolute_loc.T if transposed else absolute_loc

            adjusted_ori = ~head.ori if transposed else head.ori

            file.write(f"{adjusted_loc.row} {adjusted_loc.col} {adjusted_ori.value}\n")


# --- INDIVIDUAL --- #


@dataclass
class Individual:
    """The individual of the crossword population."""

    genome: set[Gene] = field(default_factory=set)
    """The collection of genes composing the genome."""
    grid: defaultdict[Loc, Cell] = field(default_factory=lambda: defaultdict(Cell))
    """The crossword grid. The grid is sparse, i.e. only cells with
    chars are present. Locations are relative to the first word
    inserted.
    """

    word_head: dict[str, WordHead] = field(default_factory=dict)
    """The word-to-head mapping."""
    word_genes: defaultdict[str, set[Gene]] = field(
        default_factory=lambda: defaultdict(set)
    )
    """The set of genes corresponding to each word."""

    @property
    def boundaries(self) -> tuple[Loc, Loc]:
        """The boundaries of the crossword grid.

        Returns:
            tuple[Loc, Loc]: Lower left and upper right corners of the grid.
        """
        min_row, max_row = 0, 0
        min_col, max_col = 0, 0

        for loc, cell in self.grid.items():
            if not cell:
                continue

            min_row, max_row = min(min_row, loc.row), max(max_row, loc.row)
            min_col, max_col = min(min_col, loc.col), max(max_col, loc.col)

        return Loc(row=min_row, col=min_col), Loc(row=max_row, col=max_col)

    @property
    def islands(self) -> Iterable[Individual]:
        """Disjointed gene islands of the individual's genome.

        Yields:
            Individual: An island in the form of a new individual.
        """
        visited = set()
        stack = []

        for entry_gene in self.genome:
            if entry_gene in visited:
                continue

            stack.append(entry_gene)
            island = []

            while stack:
                gene = stack.pop()
                if gene in visited:
                    continue

                visited.add(gene)
                island.append(gene)

                for adjacent_genes in map(
                    lambda char: self.word_genes[char.word], gene
                ):
                    stack += adjacent_genes

            yield Individual.safe_init(island)

    def __str__(self) -> str:
        """Serialize an individual into the form of a string matrix.

        Returns:
            str: The serialized individual.
        """
        serialized = ""
        prev_row, prev_col = 0, 0

        # Build a route for serialization.
        route = sorted(self._adjust_grid().items(), key=lambda item: item[0])

        for loc, char in route:
            if loc.row != prev_row:
                serialized += "\n"

                prev_row, prev_col = loc.row, 0

            # Padding.
            serialized += " " * (loc.col - prev_col)
            # Char.
            serialized += char

            prev_col = loc.col + 1

        return serialized

    def extend_genome(self, gene: Gene) -> None:
        """Add a new adjacent gene to the genome.

        Args:
            gene (Gene): The gene to add. Must be adjacent.
        """
        if gene in self.genome:
            return

        a_head = self.word_head.get(gene.a.word, None)
        b_head = self.word_head.get(gene.b.word, None)

        # Words are within the individual, but cross in different way.
        assert not a_head or not b_head, "Non-insertable gene"

        if not a_head and not b_head:
            # The set of genome and gene components are disjointed.
            assert not self.genome, "Non-insertable gene"

            # The genome is empty and the initial gene needs to be
            # inserted into the individual.
            return self._init_genome(gene)

        # Preliminary insertion checks.
        orientor, shift, target = (
            (a_head, gene.a.index, gene.b) if a_head else (b_head, gene.b.index, gene.a)
        )
        assert self._preinsertion(orientor, shift, target), "Non-insertable gene"

        # Insertion.
        self._insertion(orientor, shift, target)

    def shrink_genome(self, gene: Gene) -> None:
        """Removes a gene from the genome.

        Args:
            gene (Gene): The gene to remove.
        """
        assert gene in self.genome, "The gene is not in the genome"

        for head in map(lambda char: self.word_head[char.word], gene):
            cursor = head.loc
            while True:
                match head.ori:
                    case Ori.HOR:
                        if self.grid[cursor].hor is None:
                            break

                        # Uncross the orthogonal char.
                        if orthogonal_char := self.grid[cursor].ver:
                            self._uncross(self.grid[cursor].hor, orthogonal_char)

                        self.grid[cursor].hor = None
                    case Ori.VER:
                        if self.grid[cursor].ver is None:
                            break

                        # Uncross the orthogonal char.
                        if orthogonal_char := self.grid[cursor].hor:
                            self._uncross(self.grid[cursor].ver, orthogonal_char)

                        self.grid[cursor].ver = None

                cursor = Loc(
                    row=cursor.row + head.ori.drow, col=cursor.col + head.ori.dcol
                )

            self.word_head.pop(head.word)

    @classmethod
    def safe_init(cls, genome: Iterable[Gene]) -> Individual:
        """Initialize an individual with a given gene sequence.

        Args:
            genome (Iterable[Gene]): The gene sequence to initialize
                the individual with. Must be adjacent.

        Returns:
            Individual: The initialized individual.
        """
        individual = cls()

        for gene in genome:
            individual.extend_genome(gene)

        return individual

    def _adjust_grid(self) -> dict[Loc, str]:
        """Make the coordinates non-negative and try to fit within the
        size constraints of the grid.

        Returns:
            dict[Loc, str]: Adjusted preserialized grid.
        """
        origin, _ = self.boundaries

        adjusted_grid = {}

        for loc, cell in self.grid.items():
            if not cell:
                continue

            adjusted_loc = Loc.transform(loc, origin)
            char = cell.hor or cell.ver

            adjusted_grid[adjusted_loc] = char.value

        return adjusted_grid

    def _init_genome(self, gene: Gene) -> None:
        """Initialize the genome with the first gene.

        Args:
            gene (Gene): The gene to initialize the genome with.
        """
        self.genome.add(gene)

        # The A-component of the gene.
        for i in range(len(gene.a.word)):
            loc = Loc(row=0, col=i)

            self.grid[loc].hor = Char(word=gene.a.word, index=i)

            if i == 0:
                self.word_head[gene.a.word] = WordHead(
                    word=gene.a.word, loc=loc, ori=Ori.HOR
                )

        self.word_genes[gene.a.word].add(gene)

        # The B-component of the gene.
        for i in range(len(gene.b.word)):
            loc = Loc(row=-gene.b.index + i, col=gene.a.index)

            self.grid[loc].ver = Char(word=gene.b.word, index=i)

            if i == 0:
                self.word_head[gene.b.word] = WordHead(
                    word=gene.b.word, loc=loc, ori=Ori.VER
                )

        self.word_genes[gene.b.word].add(gene)

    def _preinsertion(self, orientor: WordHead, shift: int, target: Char) -> bool:
        """Check if the gene can be inserted into the individual.

        Args:
            orientor (WordHead): Word head (one of the gene components)
                that is already on the crossword grid.
            shift (int): Shift in direction relative to a word already
                inserted in the crossword grid.
            target (Char): The word (gene component) to insert.

        Returns:
            bool: True if the gene can be inserted, False otherwise.
        """
        end_i = len(target.word) - 1

        for i, char in enumerate(target.word):
            match orientor.ori:
                case Ori.HOR:
                    loc = Loc(
                        row=orientor.loc.row - target.index + i,
                        col=orientor.loc.col + shift,
                    )

                    # Check if the cell is already occupied in the considered orientation.
                    if self.grid[loc].ver is not None:
                        return False

                    # Check the cell above the word.
                    if i == 0 and self.grid[Loc(row=loc.row - 1, col=loc.col)]:
                        return False

                    if (orthogonal_char := self.grid[loc].hor) is None:
                        # Check the cells to the left and right of the word.
                        for adjacent_loc in (
                            Loc(row=loc.row, col=loc.col - 1),
                            Loc(row=loc.row, col=loc.col + 1),
                        ):
                            if self.grid[adjacent_loc]:
                                return False
                    elif orthogonal_char.value != char:
                        return False

                    # Check the cell below the word.
                    if i == end_i and self.grid[Loc(row=loc.row + 1, col=loc.col)]:
                        return False
                case Ori.VER:
                    loc = Loc(
                        row=orientor.loc.row + shift,
                        col=orientor.loc.col - target.index + i,
                    )

                    # Check if the cell is already occupied in the considered orientation.
                    if self.grid[loc].hor is not None:
                        return False

                    # Check the cell to the left of the word.
                    if i == 0 and self.grid[Loc(row=loc.row, col=loc.col - 1)]:
                        return False

                    if (orthogonal_char := self.grid[loc].ver) is None:
                        # Check the cells above and below the word.
                        for adjacent_loc in (
                            Loc(row=loc.row - 1, col=loc.col),
                            Loc(row=loc.row + 1, col=loc.col),
                        ):
                            if self.grid[adjacent_loc]:
                                return False
                    elif orthogonal_char.value != char:
                        return False

                    # Check the cell to the right of the word.
                    if i == end_i and self.grid[Loc(row=loc.row, col=loc.col + 1)]:
                        return False

        return True

    def _cross(self, a: Char, b: Char) -> None:
        """Add a new gene to the individual's genome.

        Args:
            a (Char): The A-component of the gene.
            b (Char): The B-component of the gene.
        """
        gene = Gene.safe_init(a=a, b=b)

        self.genome.add(gene)

        self.word_genes[a.word].add(gene)
        self.word_genes[b.word].add(gene)

    def _insertion(self, orientor: WordHead, shift: int, target: Char) -> None:
        """Insert the gene into the individual's crossword grid.

        Args:
            orientor (WordHead): Word head (one of the gene components)
                that is already on the crossword grid.
            shift (int): Shift in direction relative to a word already
                inserted in the crossword grid.
            target (Char): The word (gene component) to insert.
        """
        for i in range(len(target.word)):
            match orientor.ori:
                case Ori.HOR:
                    char = Char(word=target.word, index=i)

                    loc = Loc(
                        row=orientor.loc.row - target.index + i,
                        col=orientor.loc.col + shift,
                    )

                    # Genome actualization.
                    if orthogonal_char := self.grid[loc].hor:
                        self._cross(char, orthogonal_char)

                    # Grid filling.
                    self.grid[loc].ver = char

                    # Word head registration.
                    if i == 0:
                        self.word_head[target.word] = WordHead(
                            word=target.word, loc=loc, ori=~orientor.ori
                        )
                case Ori.VER:
                    char = Char(word=target.word, index=i)

                    loc = Loc(
                        row=orientor.loc.row + shift,
                        col=orientor.loc.col - target.index + i,
                    )

                    # Genome actualization.
                    if orthogonal_char := self.grid[loc].ver:
                        self._cross(char, orthogonal_char)

                    # Grid filling.
                    self.grid[loc].hor = char

                    # Word head registration.
                    if i == 0:
                        self.word_head[target.word] = WordHead(
                            word=target.word, loc=loc, ori=~orientor.ori
                        )

    def _uncross(self, a: Char, b: Char) -> None:
        """Remove a gene from the individual's genome.

        Args:
            a (Char): The A-component of the gene.
            b (Char): The B-component of the gene.
        """
        gene = Gene.safe_init(a=a, b=b)

        self.genome.remove(gene)

        self.word_genes[a.word].remove(gene)
        self.word_genes[b.word].remove(gene)


# --- GENETIC ALGORITHM --- #


def initialize_population(context: Context) -> list[Individual]:
    """Initialize the population of the crossword generation.

    Args:
        context (Context): The context of crossword generation.

    Returns:
        list[Individual]: The initial population.
    """
    return [Individual.safe_init({gene}) for gene in context.genes]


def crossover(mother: Individual, father: Individual) -> Individual:
    """Crossover of two individuals to produce an offspring.

    Args:
        mother (Individual): The mother (first parent).
        father (Individual): The father (second parent).

    Returns:
        Individual: The offspring.
    """
    offspring = copy(mother)

    for word, genes in father.word_genes.items():
        if word in mother.word_head:
            # Saturation of the offspring's genome (same as the
            # mother's genome) with the father's genes.
            for gene in genes:
                with suppress(AssertionError):
                    offspring.extend_genome(gene)

    return offspring


def mutate(context: Context, individual: Individual) -> None:
    """Mutate the individual's genome.

    Args:
        context (Context): The context of crossword generation.
        individual (Individual): The individual to mutate.
    """
    for _ in range(MUTATION_ATTEMPTS):
        if random() < GENOME_EXTENSION_RATE:
            target_genes = set()

            for word, genes in context.word_genes.items():
                if word in individual.word_head:
                    target_genes |= genes

            target_genes -= individual.genome

            if not target_genes:
                continue

            gene = choice(tuple(target_genes))

            with suppress(AssertionError):
                individual.extend_genome(gene)
        elif random() < GENOME_SHRINKAGE_RATE:
            if not individual.genome:
                continue

            gene = choice(tuple(individual.genome))

            individual.shrink_genome(gene)


def valid_size(individual: Individual) -> bool:
    """Check if the size of the individual is valid. The function takes
    into account the transposed grid.

    Args:
        individual (Individual): The individual to check.

    Returns:
        bool: True if the size is valid, False otherwise.
    """
    min_loc, max_loc = individual.boundaries

    row_size = max_loc.row - min_loc.row + 1
    col_size = max_loc.col - min_loc.col + 1

    return (
        row_size <= ROW_SIZE
        and col_size <= COL_SIZE
        or row_size <= COL_SIZE
        and col_size <= ROW_SIZE
    )


def fitness(individual: Individual) -> float:
    """Evaluate the individual's fitness.

    Args:
        individual (Individual): The individual to evaluate.

    Returns:
        float: The individual's fitness.
    """
    if not valid_size(individual):
        return 0

    island_count = len(tuple(individual.islands))

    if island_count == 0:
        return 0

    return -len(individual.word_head) / island_count


def evolve_population(
    context: Context, prev_step: list[Individual]
) -> list[Individual]:
    """Evolve the population of the crossword generation.

    Args:
        context (Context): The context of crossword generation.
        prev_step (list[Individual]): The previous population.

    Returns:
        list[Individual]: The evolved population.
    """
    population = list(prev_step)

    alpha_partner = prev_step[0]
    beta_partners = sample(prev_step, k=len(context.words))

    for beta_partner in beta_partners:
        offspring = crossover(alpha_partner, beta_partner)

        mutate(context, offspring)

        population.append(offspring)

    population.sort(key=fitness)

    return population[:POPULATION_LIMIT]


def generate(context: Context) -> Individual:
    """Generate the crossword.

    Args:
        context (Context): The context of crossword generation.

    Returns:
        Individual: The generated crossword.
    """
    population = initialize_population(context)

    while True:
        # Evolutionary step.
        population = evolve_population(context, population)

        # Check if the population contains a valid individual.
        predicate = next(
            filter(
                lambda individual: len(individual.word_head) == len(context.words)
                and valid_size(individual),
                population,
            ),
            None,
        )

        if predicate:
            return predicate


def main() -> None:
    # Initialize the context.
    context = Context.from_file("words.txt")

    # Generate the crossword.
    individual = generate(context)

    # Output the crossword.
    output_individual(context, "output.txt", individual)

    print(individual)


if __name__ == "__main__":
    main()
