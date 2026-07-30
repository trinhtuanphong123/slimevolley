import slimevolleygym.slimevolley
import slimevolleygym.mlp
from slimevolleygym.slimevolley import *

# Best-effort check that the installed Gymnasium is within stable-baselines3's
# supported range (the two libraries drift independently). Emits at most a
# warning; never breaks import.
try:
    from slimevolleygym._compat import verify_compatibility
    verify_compatibility()
except Exception:
    pass

