"""
@Time ： 2023/7/21 18:55
@Auth ： yangyu
@File ：__init__.py.py
@Motto：ABC(Always Be Coding)
"""

is_sample_core = False

if is_sample_core:
    from descalefed.core_simple import Variable
    from descalefed.core_simple import Function
    from descalefed.core_simple import using_config
    from descalefed.core_simple import no_grad
    from descalefed.core_simple import as_array
    from descalefed.core_simple import as_variable
    from descalefed.core_simple import setup_variable
    # from dezero.utils import get_dot_graph
    # from dezero.plt_dot_graph import plt_dot_graph
else:
    from descalefed.core import Variable
    from descalefed.core import Function
    from descalefed.core import using_config
    from descalefed.core import no_grad
    from descalefed.core import as_array
    from descalefed.core import as_variable
    from descalefed.core import setup_variable

import descalefed.functions
import descalefed.utils

setup_variable()
