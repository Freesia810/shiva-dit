import sys
import os

extern_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "extern", "OmniGen2")
if extern_path not in sys.path:
    sys.path.append(extern_path)