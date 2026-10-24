"""Exception hierarchy for the package.

Every error raised on purpose derives from `ConstellationError`. The schema,
feature and config errors also derive from `ValueError`, so callers that
already catch `ValueError` keep working.
"""


class ConstellationError(Exception):
    """Base class for errors raised by Constellation."""


class ConfigError(ConstellationError, ValueError):
    """A configuration file or value is invalid."""


class SchemaError(ConstellationError, ValueError):
    """A data frame does not satisfy its column contract."""


class FeatureMismatchError(ConstellationError, ValueError):
    """Input features differ from the features a model was trained on."""


class StoreError(ConstellationError):
    """The telemetry store could not complete an operation."""


class ManifestError(ConstellationError):
    """Files on disk differ from the manifest that describes them."""


class ArtifactError(ConstellationError):
    """A model artifact is missing, malformed or cannot be written."""
