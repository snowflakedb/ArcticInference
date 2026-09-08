"""Tiny HTTP fetcher with retries and an optional on-disk cache.

The NVIDIA docs site is served behind a CDN that occasionally returns 5xx
under load. A short retry loop with linear back-off makes the recovery much
more reliable when stitching hundreds of pages.

404 responses are surfaced as :class:`NotFound` (no retry) so callers can
distinguish "this URL genuinely does not exist" (e.g. a missing
``.html.md`` mirror for a single page on an otherwise-mirrored site) from
"the request failed transiently".
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class NotFound(RuntimeError):
    """Raised on HTTP 404 — the URL doesn't exist (don't retry)."""

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


class Fetcher:
    """Fetch URLs as text with retries and an optional file cache.

    The cache is keyed by the SHA-1 of the URL; a successful fetch is written
    atomically (``.tmp`` then ``rename``) so concurrent runs do not see a
    partially-written file. The cache stores the body only, not the response
    headers — this is fine for the static docs we deal with.
    """

    def __init__(
        self,
        *,
        retries: int = 3,
        sleep_s: float = 1.0,
        timeout_s: float = 60.0,
        cache_dir: Path | None = None,
    ) -> None:
        self.retries = retries
        self.sleep_s = sleep_s
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / digest

    def get(self, url: str, *, allow_empty: bool = True) -> str:
        """Return the body of ``url`` as text.

        If ``allow_empty`` is ``False`` an empty body is treated as a failure
        and not cached, which is what the new-style mirror path wants when it
        wants to detect the silently-empty ``.html.md`` files served for
        old-style docs.
        """

        cache_path = self._cache_path(url)
        if cache_path is not None and cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
            if text or allow_empty:
                return text

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
                with urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    charset = resp.headers.get_content_charset() or "utf-8"
                    text = raw.decode(charset, errors="replace")
                if cache_path is not None and (text or allow_empty):
                    tmp = cache_path.with_suffix(".tmp")
                    tmp.write_text(text, encoding="utf-8")
                    tmp.replace(cache_path)
                return text
            except HTTPError as e:
                if e.code == 404:
                    raise NotFound(f"404 Not Found: {url}") from e
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.sleep_s * attempt)
            except (URLError, TimeoutError) as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.sleep_s * attempt)
        raise RuntimeError(f"Failed to fetch {url}: {last_err}")
