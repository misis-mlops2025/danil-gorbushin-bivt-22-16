from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

app = typer.Typer()


@app.command()
def main(input_path: Path, output_path: Path):
    logger.info("plots: in=%s out=%s", input_path, output_path)
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
