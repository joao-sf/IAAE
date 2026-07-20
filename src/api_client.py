from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)


class ComprasAPIError(RuntimeError):
    """Erro de comunicação ou interpretação da API Compras.gov.br."""


def extract_records(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    embedded = payload.get("_embedded")
    if isinstance(embedded, dict):
        for value in embedded.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    for key in (
        "resultado",
        "content",
        "results",
        "resultados",
        "data",
        "items",
        "materiais",
        "licitacoes",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


class ComprasAPIClient:
    def __init__(
        self,
        timeout_seconds: float = 45,
        delay_seconds: float = 0.20,
        page_size: int = 500,
        max_pages: int = 500,
    ) -> None:
        if not 10 <= page_size <= 500:
            raise ValueError("page_size deve estar entre 10 e 500.")
        if max_pages <= 0:
            raise ValueError("max_pages deve ser maior que zero.")

        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds
        self.page_size = page_size
        self.max_pages = max_pages

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "IAAE/1.0 (projeto de dados públicos)",
            }
        )
        retry_policy = Retry(
            total=4,
            connect=4,
            read=4,
            status=4,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_policy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        LOGGER.debug("GET %s params=%s", url, params)
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise ComprasAPIError(
                f"A resposta recebida não é um JSON válido: {response.url}"
            ) from exc

    def paginate(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        page_param: str = "pagina",
        page_size_param: str = "tamanhoPagina",
        first_page: int = 1,
    ) -> Iterable[tuple[int, Any, list[dict[str, Any]]]]:
        base_params = dict(params or {})
        seen_fingerprints: set[str] = set()

        for page_number in range(first_page, first_page + self.max_pages):
            request_params = {
                **base_params,
                page_param: page_number,
                page_size_param: self.page_size,
            }
            payload = self.get_json(url, request_params)
            records = extract_records(payload)

            if not records:
                LOGGER.info("Nenhum registro encontrado na página %s.", page_number)
                break

            serialized = json.dumps(
                records,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            if fingerprint in seen_fingerprints:
                LOGGER.warning(
                    "Página repetida detectada em %s. Paginação interrompida.",
                    url,
                )
                break
            seen_fingerprints.add(fingerprint)

            LOGGER.info("Página %s: %s registro(s).", page_number, len(records))
            yield page_number, payload, records

            total_pages = None
            if isinstance(payload, dict):
                total_pages = (
                    payload.get("totalPaginas")
                    or payload.get("totalPages")
                    or payload.get("total_pages")
                )

            if total_pages is not None:
                try:
                    if page_number >= int(total_pages):
                        break
                except TypeError, ValueError:
                    LOGGER.warning("Valor inválido em totalPaginas: %r", total_pages)
            elif len(records) < self.page_size:
                break

            time.sleep(self.delay_seconds)
        else:
            LOGGER.warning(
                "Limite de %s páginas atingido em %s.",
                self.max_pages,
                url,
            )

    @staticmethod
    def save_raw_payload(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
