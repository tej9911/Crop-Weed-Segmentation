"""Runtime settings, read from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'models'
DEFAULT_MODEL_NAMES = ('mt_ahnet_model.keras', 'mt_ahnet_model.h5')


def load_dotenv(path=PROJECT_ROOT / '.env'):
    """Load KEY=VALUE lines from .env without overriding variables already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == '':
        return default
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'off'):
        return False
    raise RuntimeError(f'{name} must be true or false, got {value!r}.')


def _env_int(name, default, minimum=1):
    value = os.environ.get(name)
    if value is None or value.strip() == '':
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f'{name} must be an integer, got {value!r}.') from exc
    if parsed < minimum:
        raise RuntimeError(f'{name} must be at least {minimum}.')
    return parsed


def _resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    environment: str = 'development'
    allow_demo_mode: bool = True
    model_path: Path | None = None
    data_dir: Path = PROJECT_ROOT / 'data' / 'dataset'
    max_upload_bytes: int = 10 * 1024 * 1024
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    low_confidence_threshold: float = 0.5

    @property
    def is_production(self):
        return self.environment == 'production'

    @property
    def demo_mode_enabled(self):
        """Demo mode is never allowed in production."""
        return self.allow_demo_mode and not self.is_production

    @property
    def metadata_path(self):
        if self.model_path is None:
            return None
        return self.model_path.with_suffix('.metadata.json')

    @classmethod
    def from_env(cls):
        environment = os.environ.get('APP_ENV', 'development').strip().lower() or 'development'
        is_production = environment == 'production'

        configured_model = os.environ.get('MODEL_PATH', '').strip()
        if configured_model:
            model_path = _resolve(configured_model)
        else:
            model_path = next(
                (DEFAULT_MODEL_DIR / name for name in DEFAULT_MODEL_NAMES if (DEFAULT_MODEL_DIR / name).exists()),
                DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAMES[0],
            )

        origins = os.environ.get('CORS_ORIGINS', '').strip()
        return cls(
            environment=environment,
            allow_demo_mode=_env_bool('ALLOW_DEMO_MODE', not is_production),
            model_path=model_path,
            data_dir=_resolve(os.environ.get('DATA_DIR', '').strip() or 'data/dataset'),
            max_upload_bytes=_env_int('MAX_UPLOAD_BYTES', 10 * 1024 * 1024),
            cors_origins=tuple(o.strip() for o in origins.split(',') if o.strip()),
        )
