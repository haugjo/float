class Event(list):
    """
    Base class of the event-driven training procedure. Events are created and triggered by the Pipeline class and
    then call functions of the provided objects.
    """
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)
