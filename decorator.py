from functools import wraps


def catch_and_log(logger,return_default=None,reraise=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            try:
                return func(*args,**kwargs)
            except Exception as e:
                logger.error(e,stacklevel=2)
                if reraise:
                    raise
                else:
                    return return_default

        return wrapper
    return decorator