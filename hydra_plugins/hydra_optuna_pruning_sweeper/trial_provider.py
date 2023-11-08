"""
Singleton used to pass the current ``Trial`` object to the objective function.
"""

from typing import Optional
from optuna import Trial

trial: Optional[Trial] = None
