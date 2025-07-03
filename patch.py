import inspect

# Add getargspec if it doesn't exist (it was removed in Python 3.11)
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec