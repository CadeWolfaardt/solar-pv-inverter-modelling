def running_in_ipython_kernel():
    """Determine if current session is an IPython Kernel."""
    try:
        from IPython.core.getipython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return (
            "zmqshell" in str(type(ip)).lower() 
            or "terminalinteractiveshell" in str(type(ip)).lower()
        )
    except Exception:
        return False