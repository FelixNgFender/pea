import enum
import logging

logger = logging.getLogger(__name__)


class Type(enum.StrEnum):
    I_JEPA = enum.auto()
