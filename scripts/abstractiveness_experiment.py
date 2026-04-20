import argparse
import logging
import pathlib
from typing import Dict, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)