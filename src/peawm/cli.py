import logging
import os
import shutil

import pydantic_settings as ps
import rich.logging
import rich.prompt

from peawm import model, settings

logger = logging.getLogger(__name__)


def configure_logging(log_settings: settings.Log) -> None:
    class DistRankFilter(logging.Filter):
        """Only allows logs from rank 0 (master process) in DDP training."""

        def filter(self, record: logging.LogRecord) -> bool:  # noqa: ARG002
            rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
            # not in ddp, allow all
            if rank is None:
                return True
            # only allow logs from rank 0
            return int(rank) == 0

    logging.basicConfig(
        level=logging.DEBUG if log_settings.verbose else logging.INFO,
        format="%(message)s",
        handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
    )
    # add rank filter to root logger handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(DistRankFilter())
    logger.debug("running with settings %s", log_settings)


class Clean(settings.Clean):
    """Cleans training artifacts."""

    def cli_cmd(self) -> None:
        configure_logging(self)

        model_artifact_dirs = [self.ckpt_dir / model_type for model_type in model.Type]
        if self.force or rich.prompt.Confirm.ask(
            f"delete all model artifact directories: {', '.join(str(d) for d in model_artifact_dirs)}? THIS ACTION "
            "CANNOT BE UNDONE.",
            default=False,
        ):
            for dir_path in model_artifact_dirs:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    logger.info("deleted directory: %s", dir_path)
                else:
                    logger.info("directory does not exist, skipping: %s", dir_path)


class Command(
    ps.BaseSettings,
    cli_parse_args=True,
    cli_use_class_docs_for_groups=True,
    cli_kebab_case=True,
):
    """CLI for playing with kindergartener world models."""

    clean: ps.CliSubCommand[Clean]

    def cli_cmd(self) -> None:
        ps.CliApp.run_subcommand(self)


def main() -> None:
    ps.CliApp.run(Command)


if __name__ == "__main__":
    main()
