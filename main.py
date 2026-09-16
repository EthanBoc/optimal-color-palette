"""Command-line entry point.

Examples
--------
    python main.py images/lyon.png -k 100 --method median-cut
    python main.py images/lyon.png -k 100 --method k-means --space lab --blur gauss3
    python main.py images/lyon.png -k 100 --benchmark
"""

import argparse

from palette import pil_quantize, quantize
from palette.colour_spaces import SPACES
from palette.methods import METHODS
from palette.quantize import benchmark


def main():
    parser = argparse.ArgumentParser(description="Reduce an image to a k-colour palette.")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("-k", type=int, default=100, help="palette size (default: 100)")
    parser.add_argument("--method", choices=sorted(METHODS), default="median-cut")
    parser.add_argument("--space", choices=SPACES, default="rgb", help="colour-matching space")
    parser.add_argument("--blur", choices=["gauss3", "gauss7"], help="optional blur pre-pass")
    parser.add_argument("--seed", type=int, default=0, help="k-means seed, for reproducibility")
    parser.add_argument("-o", "--output", help="where to write the recoloured image")
    parser.add_argument("--benchmark", action="store_true", help="compare every method instead")
    args = parser.parse_args()

    if args.benchmark:
        rows = benchmark(args.image, args.k, seed=args.seed)
        _, _, pil_error, pil_time = pil_quantize(args.image, args.k)
        rows.append({"method": "PIL quantize", "space": "rgb", "blur": "none",
                     "error": round(pil_error, 2), "seconds": round(pil_time, 2)})

        print(f"{'method':<14}{'space':<7}{'blur':<8}{'error':>8}{'seconds':>10}")
        for row in rows:
            print(f"{row['method']:<14}{row['space']:<7}{row['blur']:<8}"
                  f"{row['error']:>8.2f}{row['seconds']:>10.2f}")
        return

    image, palette, _, error = quantize(
        args.image, args.k, args.method, args.space, args.blur, args.seed
    )
    print(f"method={args.method} space={args.space} blur={args.blur or 'none'} "
          f"k={len(palette)} mean_error={error:.2f}")

    output = args.output or f"quantized_{args.method}_k{args.k}.png"
    image.save(output)
    print(f"written to {output}")


if __name__ == "__main__":
    main()
