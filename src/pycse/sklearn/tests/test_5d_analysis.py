"""Analyze 5D padding transformation pattern."""

import numpy as np


def analyze_5d_with_permutation():
    """Show how Y position changes with 5D padding and permutation."""

    n_layers = 10
    perm = np.array([1, 2, 3, 4, 0])  # Rotation for 5D
    inv_perm = np.argsort(perm)  # [4, 0, 1, 2, 3]

    print("=" * 70)
    print("CINN with 5D Padding and Permutation")
    print("=" * 70)
    print("\n5D layout: [pad0, pad1, Y, pad3, pad4]")
    print("Y starts at position 2 (middle)")
    print("Permutation: [0,1,2,3,4] -> [1,2,3,4,0] (rotation)")
    print(f"Inverse permutation: {inv_perm}")
    print("\n" + "-" * 70)

    # Track Y position through the layers
    y_position = 2  # Y starts at position 2

    for layer_idx in range(n_layers):
        # Determine split for 5D
        if layer_idx % 2 == 0:
            split_idx = 5 // 2  # = 2
        else:
            split_idx = 5 - 5 // 2  # = 3

        # Check if Y gets transformed
        y_transformed = y_position >= split_idx

        unchanged = f"[0:{split_idx}]"
        transform = f"[{split_idx}:5]"

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
    y_position = 2
    transform_count = 0
    for layer_idx in range(n_layers):
        if layer_idx % 2 == 0:
            split_idx = 2
        else:
            split_idx = 3

        if y_position >= split_idx:
            transform_count += 1

        if layer_idx < n_layers - 1:
            y_position = int(perm[y_position])

    print("\n" + "=" * 70)
    print(f"✓ Y gets transformed in {transform_count}/{n_layers} layers!")
    print(f"   That's {100 * transform_count / n_layers:.0f}% coverage")
    print("=" * 70)

    # Show position sequence
    y_position = 2
    positions = [y_position]
    for i in range(n_layers - 1):
        y_position = int(perm[y_position])
        positions.append(y_position)

    print(f"\nY position sequence: {positions}")
    print(f"Unique positions visited: {sorted(set(positions))}")


if __name__ == "__main__":
    analyze_5d_with_permutation()
