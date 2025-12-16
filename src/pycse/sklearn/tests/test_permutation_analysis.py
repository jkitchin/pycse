"""Analyze how permutation improves dimension mixing."""

import numpy as np


def analyze_with_permutation():
    """Show how Y position changes with permutation [0,1,2] -> [1,2,0]."""

    n_layers = 8
    perm = np.array([1, 2, 0])  # Rotation
    inv_perm = np.argsort(perm)  # [2, 0, 1]

    print("=" * 70)
    print("CINN with Permutation Layers (FIXED)")
    print("=" * 70)
    print("\n3D layout: [padding_0, Y, padding_2]")
    print("Permutation: [0,1,2] -> [1,2,0] (rotation)")
    print(f"Inverse permutation: {inv_perm}")
    print("\n" + "-" * 70)

    # Track Y position through the layers
    y_position = 1  # Y starts at position 1

    for layer_idx in range(n_layers):
        # Determine split
        if layer_idx % 2 == 0:
            split_idx = 3 // 2  # = 1
        else:
            split_idx = 3 - 3 // 2  # = 2

        # Check if Y gets transformed
        y_transformed = y_position >= split_idx

        unchanged = f"[0:{split_idx}]"
        transform = f"[{split_idx}:3]"

        status = "TRANSFORMED" if y_transformed else "UNCHANGED"
        print(
            f"Layer {layer_idx}: Y at pos {y_position}, split={split_idx}, "
            f"unchanged={unchanged}, transform={transform} -> Y {status}"
        )

        # Apply permutation (except after last layer)
        if layer_idx < n_layers - 1:
            # Find new position of Y after permutation
            y_position = int(perm[y_position])
            print(f"          After permutation: Y moves to position {y_position}")

    # Count transformations
    y_position = 1
    transform_count = 0
    for layer_idx in range(n_layers):
        if layer_idx % 2 == 0:
            split_idx = 1
        else:
            split_idx = 2

        if y_position >= split_idx:
            transform_count += 1

        if layer_idx < n_layers - 1:
            y_position = int(perm[y_position])

    print("\n" + "=" * 70)
    print(f"✓ Y gets transformed in {transform_count}/{n_layers} layers!")
    print("=" * 70)


def analyze_without_permutation():
    """Show the old behavior without permutation."""
    print("\n\n" + "=" * 70)
    print("CINN without Permutation (OLD - BROKEN)")
    print("=" * 70)
    print("\n3D layout: [padding_0, Y, padding_2]")
    print("No permutation - Y stays at position 1\n")
    print("-" * 70)

    n_layers = 8
    y_position = 1  # Y always at position 1

    transform_count = 0
    for layer_idx in range(n_layers):
        if layer_idx % 2 == 0:
            split_idx = 1
        else:
            split_idx = 2

        y_transformed = y_position >= split_idx
        if y_transformed:
            transform_count += 1

        unchanged = f"[0:{split_idx}]"
        transform = f"[{split_idx}:3]"
        status = "TRANSFORMED" if y_transformed else "UNCHANGED"

        print(
            f"Layer {layer_idx}: Y at pos {y_position}, split={split_idx}, "
            f"unchanged={unchanged}, transform={transform} -> Y {status}"
        )

    print("\n" + "=" * 70)
    print(f"❌ Y only gets transformed in {transform_count}/{n_layers} layers")
    print("=" * 70)


if __name__ == "__main__":
    analyze_with_permutation()
    analyze_without_permutation()
