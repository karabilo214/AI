"""Адаптеры каталога: метаданные провайдера → `CatalogEntry` (4.2 MOD-01).

Каждый адаптер читает только открытые метаданные из списка источников
(`sources.py`), ничего не скачивает и ничего не знает об интерфейсе.

- Hugging Face: `GET /api/models/<repo>?blobs=true` — ревизия (коммит), лицензия,
  архитектура и контекст из GGUF, размер и SHA-256 каждого файла. Ссылка на файл
  привязана к ревизии: `/<repo>/resolve/<коммит>/<файл>`.
- Реестр Ollama: манифест тега (версия = SHA-256 манифеста), конфиг (семейство,
  квант, размер модели) и текст лицензии; файл модели — слой
  `application/vnd.ollama.image.model`, его digest и есть SHA-256 файла.
"""

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.parse import quote

from ramvi_service.catalog import http
from ramvi_service.catalog.entry import CatalogEntry
from ramvi_service.catalog.support import capabilities

Fetch = Callable[..., bytes]

HF_REPO = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}")
OLLAMA_TAG = re.compile(r"(?:[a-z0-9][a-z0-9._-]{0,63}/)?[a-z0-9][a-z0-9._-]{0,63}:[\w.-]{1,127}")
QUANT = re.compile(r"[-._](I?Q\d+_[A-Z0-9_]*[A-Z0-9]|BF16|F16|F32)\.gguf$", re.IGNORECASE)
SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
SHA_HEX = re.compile(r"[0-9a-f]{64}")
MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
MODEL_LAYER = "application/vnd.ollama.image.model"
LICENSE_LAYER = "application/vnd.ollama.image.license"


class Provider(Protocol):
    """Источник каталога: имя и записи. Ошибка сети — исключение OSError/ValueError."""

    name: str

    def entries(self) -> list[CatalogEntry]: ...


def _json(fetch: Fetch, url: str, headers: Mapping[str, str] | None = None) -> Any:
    try:
        return json.loads(fetch(url, headers))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{url}: not JSON") from exc


def quantization(file_name: str) -> str | None:
    """Квант из имени файла GGUF (так называют файлы llama.cpp и Hugging Face)."""
    match = QUANT.search(file_name)
    return match.group(1).upper() if match else None


class HuggingFace:
    """Репозитории GGUF на Hugging Face."""

    name = "huggingface"

    def __init__(
        self, repos: list[str], *, base: str = "https://huggingface.co", fetch: Fetch = http.fetch
    ) -> None:
        bad = [repo for repo in repos if not HF_REPO.fullmatch(repo)]
        if bad:
            raise ValueError(f"bad Hugging Face repo: {bad}")
        self.repos, self.base, self.fetch = list(repos), base.rstrip("/"), fetch

    def entries(self) -> list[CatalogEntry]:
        found: list[CatalogEntry] = []
        for repo in self.repos:
            found.extend(self._repo(repo))
        return found

    def _repo(self, repo: str) -> list[CatalogEntry]:
        data = _json(self.fetch, f"{self.base}/api/models/{repo}?blobs=true")
        if data.get("private") or data.get("gated") or data.get("disabled"):
            return []  # без входа в аккаунт не скачать — в каталог не попадает
        revision = str(data.get("sha") or "")
        card = data.get("cardData") or {}
        license_id = str(card.get("license") or "") or next(
            (t.split(":", 1)[1] for t in data.get("tags", []) if t.startswith("license:")), ""
        )
        gguf = data.get("gguf") or {}
        architecture = str(gguf.get("architecture") or "")
        context = gguf.get("context_length")
        found = []
        for sibling in data.get("siblings", []):
            name = str(sibling.get("rfilename") or "")
            if not name.endswith(".gguf") or "/" in name or SPLIT.search(name):
                continue  # загрузчик берёт один файл в корне репозитория
            sha = str((sibling.get("lfs") or {}).get("sha256") or "")
            quant = quantization(name) or ""
            found.append(
                CatalogEntry(
                    id=f"{self.name}:{repo}/{name}",
                    provider=self.name,
                    name=f"{repo.split('/', 1)[1].removesuffix('-GGUF')} {quant}".strip(),
                    source=repo,
                    version=revision,
                    url=f"{self.base}/{repo}/resolve/{revision}/{quote(name)}",
                    page=f"{self.base}/{repo}",
                    file_name=name,
                    bytes=int(sibling.get("size") or (sibling.get("lfs") or {}).get("size") or 0),
                    checksum=f"sha256:{sha}" if SHA_HEX.fullmatch(sha) else "",
                    format="gguf",
                    architecture=architecture,
                    quantization=quant,
                    license=license_id,
                    capabilities=capabilities("gguf", architecture),
                    context_length=context if isinstance(context, int) and context > 0 else None,
                )
            )
        return found


def license_id(text: str) -> str:
    """Короткий идентификатор по тексту лицензии; неизвестная — «other»."""
    head = " ".join(text.split()[:12]).lower()
    if "apache license" in head and "2.0" in head:
        return "apache-2.0"
    if head.startswith("mit license"):
        return "mit"
    return "other" if text.strip() else ""


class OllamaRegistry:
    """Теги реестра Ollama (`модель:тег`, без пространства имён — `library`)."""

    name = "ollama"

    def __init__(
        self,
        tags: list[str],
        *,
        base: str = "https://registry.ollama.ai",
        page: str = "https://ollama.com",
        fetch: Fetch = http.fetch,
    ) -> None:
        bad = [tag for tag in tags if not OLLAMA_TAG.fullmatch(tag)]
        if bad:
            raise ValueError(f"bad Ollama tag: {bad}")
        self.tags, self.base, self.page, self.fetch = list(tags), base.rstrip("/"), page, fetch
        self._licenses: dict[str, str] = {}

    def entries(self) -> list[CatalogEntry]:
        return [entry for tag in self.tags if (entry := self._tag(tag)) is not None]

    def _tag(self, tag: str) -> CatalogEntry | None:
        model, label = tag.split(":", 1)
        repo = model if "/" in model else f"library/{model}"
        raw = self.fetch(f"{self.base}/v2/{repo}/manifests/{label}", {"Accept": MANIFEST})
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{tag}: manifest is not JSON") from exc
        layers = {layer.get("mediaType"): layer for layer in manifest.get("layers", [])}
        weights = layers.get(MODEL_LAYER)
        if weights is None:
            return None
        config = _json(self.fetch, f"{self.base}/v2/{repo}/blobs/{manifest['config']['digest']}")
        architecture = str(config.get("model_family") or "")
        digest = str(weights.get("digest") or "")
        return CatalogEntry(
            id=f"{self.name}:{tag}",
            provider=self.name,
            name=f"{tag} {config.get('file_type') or ''}".strip(),
            source=tag,
            version="sha256:" + hashlib.sha256(raw).hexdigest(),
            url=f"{self.base}/v2/{repo}/blobs/{digest}",
            page=f"{self.page}/{model if '/' in model else 'library/' + model}:{label}",
            file_name=digest.replace(":", "-"),  # так файл называет и сам Ollama
            bytes=int(weights.get("size") or 0),
            checksum=digest,
            format=str(config.get("model_format") or ""),
            architecture=architecture,
            quantization=str(config.get("file_type") or "").upper(),
            license=self._license(repo, layers.get(LICENSE_LAYER)),
            capabilities=capabilities(str(config.get("model_format") or ""), architecture),
            parameters=str(config.get("model_type") or "") or None,
        )

    def _license(self, repo: str, layer: Mapping[str, Any] | None) -> str:
        if layer is None:
            return ""
        digest = str(layer.get("digest") or "")
        if digest not in self._licenses:  # одна лицензия на все теги модели
            body = self.fetch(f"{self.base}/v2/{repo}/blobs/{digest}", limit=256 * 1024)
            self._licenses[digest] = license_id(body.decode("utf-8", "replace"))
        return self._licenses[digest]
