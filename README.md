# Static Warp Bubbles

This repository contains the implementation of the **Static Spherically-Symmetric Warp Bubble** metric described by Bolívar, Abellán, and Vasilev (2025).

It is built as an extension to [pyWarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory), using it as a core engine for metric handling and Einstein tensor solving.

## Installation

1.  **Install pyWarpFactory**:
    Ensure you have the main `warpfactory` package installed. If you have the source locally:
    ```bash
    cd ../WarpFactory-main
    pip install -e .
    ```

2.  **Install StaticWarpBubbles**:
    ```bash
    cd StaticWarpBubbles
    pip install -e .
    ```

## Google Colab

To run this toolkit in the cloud:
1.  Open [Google Colab](https://colab.research.google.com/).
2.  Upload `StaticWarpBubbles_Colab.ipynb` from this repository.
3.  Run the cells (the notebook handles installing `pyWarpFactory` and the extension automatically).

## Structure

-   `static_bubbles/`: Core Python package.
    -   `generator.py`: Generates metric from energy density $\rho(r)$.
    -   `analyzer.py`: Checks Energy Conditions analytically.
    -   `utils.py`: (Optional utility functions).

-   `examples/`: Demo scripts.
    -   `demo.py`: Generates plots of Density, Shift, and ECs.

-   `tests/`: Verification scripts.
    -   `validate.py`: Validates numerical solver against analytic types.

## Usage

Run the demo:
```bash
python examples/demo.py
```
