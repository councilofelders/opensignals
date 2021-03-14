
"""ysignals

Usage:
  ysignals [-v...]
  ysignals -h | --help
  ysignals --version
  ysignals [options] download --dir=<dir> [--recreate]

Options:
  -h --help         Show this screen.
  --version         Show version.
  --verbose=<level> Increase verbosity. [default: 1]

Commands:
    ysignals download --dir=somedirectory [--recreate]
        Download new data.

"""
import logging
from pathlib import Path
from typing import List, Optional

from docopt import docopt

from ysignals import __version__
from ysignals.data import yahoo
from ysignals.features import RSI

def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point.

    Args:
        argv (list of str, optional): Anything passed here will be treated as sys.argv[1:]
            (command-line arguments minus the entry point itself). Useful for testing.

    """
    args = docopt(__doc__, version=__version__)

    if args['--verbose'] and int(args['--verbose']) > 1:
        logging.basicConfig(level=logging.DEBUG)
    elif args['--verbose'] and int(args['--verbose']) == 0:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    if args['download']:
        yahoo.download_data(Path(args['--dir']), args['--recreate'])


if __name__ == '__main__':
    main()
