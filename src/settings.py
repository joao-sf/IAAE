from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    materials_url: str = os.getenv(
        "COMPRAS_MATERIAIS_URL",
        "https://dadosabertos.compras.gov.br/modulo-material/4_consultarItemMaterial",
    )
    practiced_prices_url: str = os.getenv(
        "COMPRAS_PRECOS_MATERIAIS_URL",
        "https://dadosabertos.compras.gov.br/modulo-pesquisa-preco/1_consultarMaterial",
    )
    timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "45"))
    request_delay_seconds: float = float(os.getenv("REQUEST_DELAY_SECONDS", "0.20"))
    page_size: int = int(os.getenv("PAGE_SIZE", "500"))
    max_pages: int = int(os.getenv("MAX_PAGES", "500"))
    csv_separator: str = "|"

    bronze_dir: Path = ROOT_DIR / "data" / "bronze"
    silver_dir: Path = ROOT_DIR / "data" / "silver"
    gold_dir: Path = ROOT_DIR / "data" / "gold"
    log_dir: Path = ROOT_DIR / "logs"

    def ensure_directories(self) -> None:
        for path in (
            self.bronze_dir,
            self.silver_dir,
            self.gold_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
