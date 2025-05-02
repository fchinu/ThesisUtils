"""Draw distributions of variables from input files and saves them to the output path."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def draw_distributions(input_files, output_path, labels):
    """
    Draw distributions of variables from input files and saves them to the output path.

    Parameters:
        input_files (list): List of input file paths.
        output_path (str): Path to the output directory where the plots will be saved.
        labels (list): List of labels for the input files.
    """
    # Read the data from the input files
    data = [pd.read_parquet(file) for file in input_files]

    if labels is None:
        labels = [f"File {i+1}" for i in range(len(data))]

    num_columns = len(data[0].columns)
    ncols = num_columns // 4 + (num_columns % 4 > 0)
    nrows = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 20))
    axes = axes.flatten()

    for file_idx, df in enumerate(data):
        for col_idx, column in enumerate(df.columns):
            ax = axes[col_idx]
            df[column].plot(
                kind='hist',
                ax=ax,
                bins=100,
                alpha=0.5,
                log=True,
                density=len(data) > 1,
                label=labels[file_idx]
            )
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel('Counts')

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw distributions of variables.")
    parser.add_argument(
        "input",
        nargs='+',
        type=str
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs='+',
        type=str,
        required=False
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        default="./distributions.pdf",
        help="Path to the output directory where the plots will be saved.",
    )
    args = parser.parse_args()

    draw_distributions(args.input, args.output, args.labels)
