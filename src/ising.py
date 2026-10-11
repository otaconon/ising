import argparse
import sys
import taichi as ti
from ising3d import Ising3d
from ising2d import Ising2d


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="ising", description="Ising model")
  common = argparse.ArgumentParser(add_help=False)
  
  parser.add_argument("--window-size", nargs=2, type=int, default=[1200, 1200], metavar=("WIDTH", "HEIGHT"))
  parser.add_argument("-t", "--temperature", type=float, default=2.0)
  
  sub = parser.add_subparsers(dest="simulation", required=True)
  p2 = sub.add_parser("2d", parents=[common])
  p2.add_argument("-d", "--dimensions", nargs=2, type=int, default=[150, 150], metavar=("N", "M"))
  p3 = sub.add_parser("3d", parents=[common])
  p3.add_argument("-d", "--dimensions", nargs=3, type=int, default=[150, 150, 150], metavar=("N", "M", "K"))
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  
  if any(v % 2 for v in args.dimensions):
    parser.error("all dimensions must be even")
  if args.temperature <= 0:
    parser.error("temperature must be positive")

  if args.simulation == "2d":
    args.dimensions = args.dimensions[:3]

  ti.init(arch=ti.gpu)
  
  if args.simulation == "3d":
    sim = Ising3d(tuple(args.window_size), tuple(args.dimensions), args.temperature)
  else:
    sim = Ising2d(tuple(args.window_size), tuple(args.dimensions), args.temperature)

  sim.run()
  
  return 0


if __name__ == "__main__":
  sys.exit(main())
