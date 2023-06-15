import flor
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'

flor.replay(['dp', 'dp_str'], where_clause=None, path='train.py')
