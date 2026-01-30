# stdlib
from typing import Any

def assign_attrs(
    obj: Any,
    *,
    allow_new: bool = False,
    **kwargs: Any,
) -> None:
    """
    Assign multiple attributes to an object in a single operation.

    This helper sets attributes on an existing object using keyword
    arguments. By default, it only allows assignment to attributes that
    already exist on the target object, helping to catch typos and
    unintended state mutation early. New attributes may be added
    explicitly by enabling ``allow_new``.

    This function is intended for glue code, adapters, configuration
    application, or deserialization scenarios where attribute assignment
    needs to be concise and explicit.

    Parameters
    ----------
    obj : Any
        Target object whose attributes will be assigned.
    allow_new : bool, default=False
        If ``False``, raise an ``AttributeError`` when attempting to set
        an attribute that does not already exist on ``obj``. If 
        ``True``, new attributes are created as needed.
    **kwargs : Any
        Mapping of attribute names to values to assign.

    Raises
    ------
    AttributeError
        If ``allow_new`` is ``False`` and an attribute name does not
        already exist on the target object.

    Notes
    -----
    - This function mutates ``obj`` in place.
    - Attribute assignment performed here bypasses static type checking
      and ``__slots__`` constraints.
    - Prefer explicit attribute assignment or dataclasses for core
      domain state; this helper is best suited for infrastructure and
      orchestration layers.
    """
    for k, v in kwargs.items():
        # Prevent accidental creation of new attributes unless 
        # explicitly allowed
        if not allow_new and not hasattr(obj, k):
            raise AttributeError(f"{obj!r} has no attribute {k!r}")
        # Assign the attribute value
        setattr(obj, k, v)

