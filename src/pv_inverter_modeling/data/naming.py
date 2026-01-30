from typing import Optional

# Column prefixes for predictions from models
PRED = 'pred'
NAIVE = 'naive'

def pred_name(
        *,
        family: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
    """
    Construct a standardized prediction column name.

    This helper builds prediction column identifiers using a consistent
    ``"pred::"``-prefixed, namespace-style convention. Optional
    components may be provided to denote model family and specific
    predictor name, allowing hierarchical naming of prediction outputs.

    Parameters
    ----------
    family : str, optional
        Optional model family or category (e.g. ``"naive"``,
        ``"hybrid"``, ``"learned"``). When provided, it is appended
        after the ``"pred::"`` prefix.
    name : str, optional
        Optional predictor or strategy identifier within the given
        family (e.g. ``"persistence"``, ``"ema"``). When provided, it
        is appended after the family component.

    Returns
    -------
    str
        A standardized prediction column name ending with ``"::"``.

    Examples
    --------
    >>> pred_name()
    'pred::'

    >>> pred_name(family='naive')
    'pred::naive::'

    >>> pred_name(family='naive', name='persistence')
    'pred::naive::persistence::'
    """
    parts = [PRED]
    if family is not None:
        parts.append(family)
    if name is not None:
        parts.append(name)
    return "::".join(parts) + "::"