from pathlib import Path
import matplotlib.pyplot as plt


class ArtifactWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def figure(self, figure, name):
        figure.savefig(self.directory / name, dpi=150, bbox_inches="tight")
        plt.close(figure)

    def tables(self, tables, *, index=False):
        for name, table in tables.items():
            table.to_csv(self.directory / f"{name}.csv", index=index)


def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)
