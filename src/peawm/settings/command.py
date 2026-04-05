import pathlib
from typing import Annotated

import pydantic
import pydantic_settings as ps

from peawm import constants
from peawm.settings import primitive


class Clean(primitive.Log):
    """Settings for the `clean` CLI subcommand."""

    ckpt_dir: Annotated[
        pathlib.Path,
        pydantic.Field(description="Model checkpoints directory to clean"),
    ] = constants.CKPT_DIR
    force: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("f", "force"),
            description="Force clean without user confirmation (DANGEROUS)",
        ),
    ] = False
