# modified from: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_param_validation.py
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
import functools
import math
from inspect import signature
from numbers import Integral
from numbers import Real
import operator
import warnings
import numpy as np

def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)


def validate_parameter_constraints(parameter_constraints, params, caller_name):
    """Validate types and values of given parameters.
    Parameters
    ----------
    parameter_constraints : dict or {"no_validation"}
        If "no_validation", validation is skipped for this parameter.
        If a dict, it must be a dictionary `param_name: list of constraints`.
        A parameter is valid if it satisfies one of the constraints from the list.
        Constraints can be:
        - an Interval object, representing a continuous or discrete range of numbers
        - the string "array-like"
        - the string "sparse matrix"
        - the string "random_state"
        - callable
        - None, meaning that None is a valid value for the parameter
        - any type, meaning that any instance of this type is valid
        - an Options object, representing a set of elements of a given type
        - a StrOptions object, representing a set of strings
        - the string "boolean"
        - the string "verbose"
        - the string "cv_object"
        - the string "missing_values"
        - a HasMethods object, representing method(s) an object must have
        - a Hidden object, representing a constraint not meant to be exposed to the user
    params : dict
        A dictionary `param_name: param_value`. The parameters to validate against the
        constraints.
    caller_name : str
        The name of the estimator or function or method that called this function.
    """
    # if len(set(parameter_constraints) - set(params)) != 0:
    #     raise ValueError(
    #         f"The parameter constraints {list(parameter_constraints)}"
    #         " contain unexpected parameters"
    #         f" {set(parameter_constraints) - set(params)}"
    #     )

    for param_name, param_val in params.items():
        # We allow parameters to not have a constraint so that third party estimators
        # can inherit from sklearn estimators without having to necessarily use the
        # validation tools.
        if param_name not in parameter_constraints:
            continue

        constraints = parameter_constraints[param_name]

        if constraints == "no_validation":
            continue

        constraints = [make_constraint(constraint) for constraint in constraints]

        for constraint in constraints:
            if constraint.is_satisfied_by(param_val):
                # this constraint is satisfied, no need to check further.
                break
        else:
            # No constraint is satisfied, raise with an informative message.

            # Ignore constraints that we don't want to expose in the error message,
            # i.e. options that are for internal purpose or not officially supported.
            constraints = [
                constraint for constraint in constraints if not constraint.hidden
            ]

            if len(constraints) == 1:
                constraints_str = f"{constraints[0]}"
            else:
                constraints_str = (
                    f"{', '.join([str(c) for c in constraints[:-1]])} or"
                    f" {constraints[-1]}"
                )

            raise ValueError(
                f"The {param_name!r} parameter of {caller_name} must be"
                f" {constraints_str}. Got {param_val!r} instead."
            )


def make_constraint(constraint):
    """Convert the constraint into the appropriate Constraint object.
    Parameters
    ----------
    constraint : object
        The constraint to convert.
    Returns
    -------
    constraint : instance of _Constraint
        The converted constraint.
    """
    if isinstance(constraint, str) and constraint == "array-like":
        return _ArrayLikes()
    if isinstance(constraint, str) and constraint == "random_state":
        return _RandomStates()
    if constraint is callable:
        return _Callables()
    if constraint is None:
        return _NoneConstraint()
    if isinstance(constraint, type):
        return _InstancesOf(constraint)
    if isinstance(constraint, (Interval, StrOptions, Options, HasMethods)):
        return constraint
    if isinstance(constraint, str) and constraint == "boolean":
        return _Booleans()
    if isinstance(constraint, str) and constraint == "verbose":
        return _VerboseHelper()
    if isinstance(constraint, str) and constraint == "missing_values":
        return _MissingValues()
    raise ValueError(f"Unknown constraint type: {constraint}")


def validate_params(parameter_constraints):
    """Decorator to validate types and values of functions and methods.
    Parameters
    ----------
    parameter_constraints : dict
        A dictionary `param_name: list of constraints`. See the docstring of
        `validate_parameter_constraints` for a description of the accepted constraints.
        Note that the *args and **kwargs parameters are not validated and must not be
        present in the parameter_constraints dictionary.
    Returns
    -------
    decorated_function : function or method
        The decorated function.
    """

    def decorator(func):
        # The dict of parameter constraints is set as an attribute of the function
        # to make it possible to dynamically introspect the constraints for
        # automatic testing.
        setattr(func, "_skl_parameter_constraints", parameter_constraints)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            func_sig = signature(func)

            # Map *args/**kwargs to the function signature
            params = func_sig.bind(*args, **kwargs)
            params.apply_defaults()

            # ignore self/cls and positional/keyword markers
            to_ignore = [
                p.name
                for p in func_sig.parameters.values()
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            to_ignore += ["self", "cls"]
            params = {k: v for k, v in params.arguments.items() if k not in to_ignore}

            validate_parameter_constraints(
                parameter_constraints, params, caller_name=func.__qualname__
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _type_name(t):
    """Convert type into human readable string."""
    module = t.__module__
    qualname = t.__qualname__
    if module == "builtins":
        return qualname
    elif t == Real:
        return "float"
    elif t == Integral:
        return "int"
    return f"{module}.{qualname}"


class _Constraint(ABC):
    """Base class for the constraint objects."""

    def __init__(self):
        self.hidden = False

    @abstractmethod
    def is_satisfied_by(self, val):
        """Whether or not a value satisfies the constraint.
        Parameters
        ----------
        val : object
            The value to check.
        Returns
        -------
        is_satisfied : bool
            Whether or not the constraint is satisfied by this value.
        """

    @abstractmethod
    def __str__(self):
        """A human readable representational string of the constraint."""


class _InstancesOf(_Constraint):
    """Constraint representing instances of a given type.
    Parameters
    ----------
    type : type
        The valid type.
    """

    def __init__(self, type):
        super().__init__()
        self.type = type

    def is_satisfied_by(self, val):
        return isinstance(val, self.type)

    def __str__(self):
        return f"an instance of {_type_name(self.type)!r}"


class _NoneConstraint(_Constraint):
    """Constraint representing the None singleton."""

    def is_satisfied_by(self, val):
        return val is None

    def __str__(self):
        return "None"


class _NanConstraint(_Constraint):
    """Constraint representing the indicator `np.nan`."""

    def is_satisfied_by(self, val):
        return isinstance(val, Real) and math.isnan(val)

    def __str__(self):
        return "numpy.nan"


class _PandasNAConstraint(_Constraint):
    """Constraint representing the indicator `pd.NA`."""

    def is_satisfied_by(self, val):
        try:
            import pandas as pd

            return isinstance(val, type(pd.NA)) and pd.isna(val)
        except ImportError:
            return False

    def __str__(self):
        return "pandas.NA"


class Options(_Constraint):
    """Constraint representing a finite set of instances of a given type.
    Parameters
    ----------
    type : type
    options : set
        The set of valid scalars.
    deprecated : set or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, type, options, *, deprecated=None):
        super().__init__()
        self.type = type
        self.options = options
        self.deprecated = deprecated or set()

        if self.deprecated - self.options:
            raise ValueError("The deprecated options must be a subset of the options.")

    def is_satisfied_by(self, val):
        return isinstance(val, self.type) and val in self.options

    def _mark_if_deprecated(self, option):
        """Add a deprecated mark to an option if needed."""
        option_str = f"{option!r}"
        if option in self.deprecated:
            option_str = f"{option_str} (deprecated)"
        return option_str

    def __str__(self):
        options_str = (
            f"{', '.join([self._mark_if_deprecated(o) for o in self.options])}"
        )
        return f"a {_type_name(self.type)} among {{{options_str}}}"


class StrOptions(Options):
    """Constraint representing a finite set of strings.
    Parameters
    ----------
    options : set of str
        The set of valid strings.
    deprecated : set of str or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, options, *, deprecated=None):
        super().__init__(type=str, options=options, deprecated=deprecated)


class Interval(_Constraint):
    """Constraint representing a typed interval.
    Parameters
    ----------
    type : {Integral, numbers.Real}
        The set of numbers in which to set the interval.
    left : float or int or None
        The left bound of the interval. None means left bound is -∞.
    right : float, int or None
        The right bound of the interval. None means right bound is +∞.
    closed : {"left", "right", "both", "neither"}
        Whether the interval is open or closed. Possible choices are:
        - `"left"`: the interval is closed on the left and open on the right.
          It is equivalent to the interval `[ left, right )`.
        - `"right"`: the interval is closed on the right and open on the left.
          It is equivalent to the interval `( left, right ]`.
        - `"both"`: the interval is closed.
          It is equivalent to the interval `[ left, right ]`.
        - `"neither"`: the interval is open.
          It is equivalent to the interval `( left, right )`.
    Notes
    -----
    Setting a bound to `None` and setting the interval closed is valid. For instance,
    strictly speaking, `Interval(Real, 0, None, closed="both")` corresponds to
    `[0, +∞) U {+∞}`.
    """

    @validate_params(
        {
            "type": [type],
            "left": [Integral, Real, None],
            "right": [Integral, Real, None],
            "closed": [StrOptions({"left", "right", "both", "neither"})],
        }
    )
    def __init__(self, type, left, right, *, closed):
        super().__init__()
        self.type = type
        self.left = left
        self.right = right
        self.closed = closed

        self._check_params()

    def _check_params(self):
        if self.type is Integral:
            suffix = "for an interval over the integers."
            if self.left is not None and not isinstance(self.left, Integral):
                raise TypeError(f"Expecting left to be an int {suffix}")
            if self.right is not None and not isinstance(self.right, Integral):
                raise TypeError(f"Expecting right to be an int {suffix}")
            if self.left is None and self.closed in ("left", "both"):
                raise ValueError(
                    f"left can't be None when closed == {self.closed} {suffix}"
                )
            if self.right is None and self.closed in ("right", "both"):
                raise ValueError(
                    f"right can't be None when closed == {self.closed} {suffix}"
                )

        if self.right is not None and self.left is not None and self.right <= self.left:
            raise ValueError(
                f"right can't be less than left. Got left={self.left} and "
                f"right={self.right}"
            )

    def __contains__(self, val):
        if np.isnan(val):
            return False

        left_cmp = operator.lt if self.closed in ("left", "both") else operator.le
        right_cmp = operator.gt if self.closed in ("right", "both") else operator.ge

        left = -np.inf if self.left is None else self.left
        right = np.inf if self.right is None else self.right

        if left_cmp(val, left):
            return False
        if right_cmp(val, right):
            return False
        return True

    def is_satisfied_by(self, val):
        if not isinstance(val, self.type):
            return False

        return val in self

    def __str__(self):
        type_str = "an int" if self.type is Integral else "a float"
        left_bracket = "[" if self.closed in ("left", "both") else "("
        left_bound = "-inf" if self.left is None else self.left
        right_bound = "inf" if self.right is None else self.right
        right_bracket = "]" if self.closed in ("right", "both") else ")"
        return (
            f"{type_str} in the range "
            f"{left_bracket}{left_bound}, {right_bound}{right_bracket}"
        )


class _ArrayLikes(_Constraint):
    """Constraint representing array-likes"""

    def is_satisfied_by(self, val):
        return _is_arraylike_not_scalar(val)

    def __str__(self):
        return "an array-like"


class _Callables(_Constraint):
    """Constraint representing callables."""

    def is_satisfied_by(self, val):
        return callable(val)

    def __str__(self):
        return "a callable"


class _RandomStates(_Constraint):
    """Constraint representing random states.
    Convenience class for
    [Interval(Integral, 0, 2**32 - 1, closed="both"), np.random.RandomState, None]
    """

    def __init__(self):
        super().__init__()
        self._constraints = [
            Interval(Integral, 0, 2**32 - 1, closed="both"),
            _InstancesOf(np.random.RandomState),
            _NoneConstraint(),
        ]

    def is_satisfied_by(self, val):
        return any(c.is_satisfied_by(val) for c in self._constraints)

    def __str__(self):
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )


class _Booleans(_Constraint):
    """Constraint representing boolean likes.
    Convenience class for
    [bool, np.bool_, Integral (deprecated)]
    """

    def __init__(self):
        super().__init__()
        self._constraints = [
            _InstancesOf(bool),
            _InstancesOf(np.bool_),
            _InstancesOf(Integral),
        ]

    def is_satisfied_by(self, val):
        # TODO(1.4) remove support for Integral.
        if isinstance(val, Integral) and not isinstance(val, bool):
            warnings.warn(
                "Passing an int for a boolean parameter is deprecated in version 1.2 "
                "and won't be supported anymore in version 1.4.",
                FutureWarning,
            )

        return any(c.is_satisfied_by(val) for c in self._constraints)

    def __str__(self):
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )


class _VerboseHelper(_Constraint):
    """Helper constraint for the verbose parameter.
    Convenience class for
    [Interval(Integral, 0, None, closed="left"), bool, numpy.bool_]
    """

    def __init__(self):
        super().__init__()
        self._constraints = [
            Interval(Integral, 0, None, closed="left"),
            _InstancesOf(bool),
            _InstancesOf(np.bool_),
        ]

    def is_satisfied_by(self, val):
        return any(c.is_satisfied_by(val) for c in self._constraints)

    def __str__(self):
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )


class _MissingValues(_Constraint):
    """Helper constraint for the `missing_values` parameters.
    Convenience for
    [
        Integral,
        Interval(Real, None, None, closed="both"),
        str,
        None,
        _NanConstraint(),
        _PandasNAConstraint(),
    ]
    """

    def __init__(self):
        super().__init__()
        self._constraints = [
            _InstancesOf(Integral),
            # we use an interval of Real to ignore np.nan that has its own constraint
            Interval(Real, None, None, closed="both"),
            _InstancesOf(str),
            _NoneConstraint(),
            _NanConstraint(),
            _PandasNAConstraint(),
        ]

    def is_satisfied_by(self, val):
        return any(c.is_satisfied_by(val) for c in self._constraints)

    def __str__(self):
        return (
            f"{', '.join([str(c) for c in self._constraints[:-1]])} or"
            f" {self._constraints[-1]}"
        )


class HasMethods(_Constraint):
    """Constraint representing objects that expose specific methods.
    It is useful for parameters following a protocol and where we don't want to impose
    an affiliation to a specific module or class.
    Parameters
    ----------
    methods : str or list of str
        The method(s) that the object is expected to expose.
    """

    @validate_params({"methods": [str, list]})
    def __init__(self, methods):
        super().__init__()
        if isinstance(methods, str):
            methods = [methods]
        self.methods = methods

    def is_satisfied_by(self, val):
        return all(callable(getattr(val, method, None)) for method in self.methods)

    def __str__(self):
        if len(self.methods) == 1:
            methods = f"{self.methods[0]!r}"
        else:
            methods = (
                f"{', '.join([repr(m) for m in self.methods[:-1]])} and"
                f" {self.methods[-1]!r}"
            )
        return f"an object implementing {methods}"


class _IterablesNotString(_Constraint):
    """Constraint representing iterables that are not strings."""

    def is_satisfied_by(self, val):
        return isinstance(val, Iterable) and not isinstance(val, str)

    def __str__(self):
        return "an iterable"


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(x) for x in lengths])


def check_1d(array, var_name):
    """
    check if the input is 1d array
    Parameters
    ----------
    array
    var_name

    Returns
    -------

    """
    if array.ndim != 1:
        raise ValueError("Expected 1D array for input {}, but got {}D".format(var_name, array.ndim))
