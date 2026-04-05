from typing import Annotated

import pydantic

from peawm import constants
from peawm.settings import base


class ModelBase(base.Base):
    """Settings common to all model type creation."""

    context_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Context size for the model"),
    ] = constants.CONTEXT_SIZE
