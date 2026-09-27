import sys

from . import main as _main, options

def main():
    sys.exit(_main.run(options.parse_argv()))

if __name__ == '__main__':
    main()
