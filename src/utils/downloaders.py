from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


VIDEO_DOWNLOAD_TIMEOUT = (20, 600)  # (connect_timeout, read_timeout)


class DownloadError(Exception):
    """Общая ошибка скачивания."""


class UnsupportedUrlError(DownloadError):
    """Ссылка не поддерживается."""


def build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )

    return session


def is_yandex_disk_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return "disk.yandex.ru" in host or "disk.360.yandex.ru" in host


def is_google_drive_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return "drive.google.com" in host


def extract_google_drive_file_id(url: str) -> str:
    parsed = urlparse(url)

    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)

    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]

    raise DownloadError(f"Не удалось извлечь file_id из Google Drive URL: {url}")


def get_yandex_disk_direct_url(
    public_url: str,
    session: Optional[requests.Session] = None,
    timeout: tuple[int, int] = VIDEO_DOWNLOAD_TIMEOUT,
) -> str:
    """
    Получает прямую ссылку на скачивание из публичной ссылки Яндекс.Диска.
    """
    own_session = session is None
    session = session or build_session()

    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

    try:
        response = session.get(
            api_url,
            params={"public_key": public_url},
            timeout=timeout,
        )
        response.raise_for_status()
    except RequestsConnectionError as e:
        raise DownloadError(
            "Не удалось подключиться к API Яндекс.Диска. "
            f"Соединение было сброшено: {e}"
        ) from e
    except RequestException as e:
        raise DownloadError(f"Ошибка запроса к API Яндекс.Диска: {e}") from e
    finally:
        if own_session:
            session.close()

    try:
        data = response.json()
    except ValueError as e:
        raise DownloadError(f"API Яндекс.Диска вернул не JSON: {response.text[:500]}") from e

    href = data.get("href")
    if not href:
        raise DownloadError(f"Не удалось получить прямую ссылку Яндекс.Диска: {data}")

    return href


def get_google_drive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def resolve_video_download_url(
    url: str,
    session: Optional[requests.Session] = None,
) -> str:
    if is_yandex_disk_url(url):
        return get_yandex_disk_direct_url(url, session=session)

    if is_google_drive_url(url):
        file_id = extract_google_drive_file_id(url)
        return get_google_drive_direct_url(file_id)

    return url


def save_response_to_file(
    response: requests.Response,
    output_path: str | Path,
    chunk_size: int = 1024 * 1024,
    desc: Optional[str] = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_size = int(response.headers.get("Content-Length", 0))
    progress_desc = desc or output.name

    with output.open("wb") as f, tqdm(
        total=total_size if total_size > 0 else None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=progress_desc,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            progress_bar.update(len(chunk))

    return output


def download_from_direct_url(
    direct_url: str,
    output_path: str | Path,
    session: Optional[requests.Session] = None,
    timeout: tuple[int, int] = VIDEO_DOWNLOAD_TIMEOUT,
    chunk_size: int = 1024 * 1024,
    headers: Optional[dict] = None,
) -> Path:
    own_session = session is None
    session = session or build_session()

    request_headers = dict(session.headers)
    if headers:
        request_headers.update(headers)

    try:
        with session.get(
            direct_url,
            stream=True,
            timeout=timeout,
            headers=request_headers,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            return save_response_to_file(
                response=response,
                output_path=output_path,
                chunk_size=chunk_size,
            )
    except RequestsConnectionError as e:
        raise DownloadError(f"Соединение было сброшено во время скачивания: {e}") from e
    except RequestException as e:
        raise DownloadError(f"Ошибка скачивания файла: {e}") from e
    finally:
        if own_session:
            session.close()


def download_google_drive_with_confirm(
    url: str,
    output_path: str | Path,
    session: Optional[requests.Session] = None,
    timeout: tuple[int, int] = VIDEO_DOWNLOAD_TIMEOUT,
    chunk_size: int = 1024 * 1024,
) -> Path:
    file_id = extract_google_drive_file_id(url)
    base_url = "https://drive.google.com/uc?export=download"

    own_session = session is None
    session = session or build_session()

    try:
        response = session.get(
            base_url,
            params={"id": file_id},
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        disposition = response.headers.get("Content-Disposition", "")

        if "attachment" in disposition.lower() or "video" in content_type.lower():
            return save_response_to_file(
                response=response,
                output_path=output_path,
                chunk_size=chunk_size,
            )

        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                confirm_token = value
                break

        if confirm_token:
            with session.get(
                base_url,
                params={"id": file_id, "confirm": confirm_token},
                stream=True,
                timeout=timeout,
            ) as confirmed_response:
                confirmed_response.raise_for_status()
                return save_response_to_file(
                    response=confirmed_response,
                    output_path=output_path,
                    chunk_size=chunk_size,
                )

        raise DownloadError(
            "Google Drive не отдал файл напрямую. "
            "Возможно, файл недоступен публично или требуется дополнительное подтверждение."
        )

    except RequestsConnectionError as e:
        raise DownloadError(f"Ошибка соединения с Google Drive: {e}") from e
    except RequestException as e:
        raise DownloadError(f"Ошибка запроса к Google Drive: {e}") from e
    finally:
        if own_session:
            session.close()


def download_video(
    url: str,
    filename: str | Path,
    timeout: tuple[int, int] = VIDEO_DOWNLOAD_TIMEOUT,
    chunk_size: int = 1024 * 1024,
    force_reload: bool = True,
) -> Path:
    
    output_path =  Path("data") / "videos" / filename

    if output_path.exists() and not force_reload:
        print(f"File already exists: {output_path}. Skipping download.")
        return output_path

    session = build_session()

    try:
        if is_yandex_disk_url(url):
            direct_url = get_yandex_disk_direct_url(
                public_url=url,
                session=session,
                timeout=timeout,
            )
            return download_from_direct_url(
                direct_url=direct_url,
                output_path=output_path,
                session=session,
                timeout=timeout,
                chunk_size=chunk_size,
            )

        if is_google_drive_url(url):
            return download_google_drive_with_confirm(
                url=url,
                output_path=output_path,
                session=session,
                timeout=timeout,
                chunk_size=chunk_size,
            )

        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return download_from_direct_url(
                direct_url=url,
                output_path=output_path,
                session=session,
                timeout=timeout,
                chunk_size=chunk_size,
            )

        raise UnsupportedUrlError(f"Неподдерживаемая ссылка: {url}")

    finally:
        session.close()

