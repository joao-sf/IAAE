from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .text_utils import normalize_text


@dataclass(frozen=True)
class Classification:
    family_code: str | None
    family_label: str | None
    score: int
    reason: str
    normalized_description: str


class MaterialClassifier:
    def __init__(self, config_path: Path) -> None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.minimum_score = int(config.get("minimum_score", 2))
        self.families: dict[str, dict[str, Any]] = config["families"]

    def classify(self, description: Any) -> Classification:
        normalized = normalize_text(description)
        if not normalized:
            return Classification(None, None, 0, "descrição vazia", normalized)

        candidates: list[tuple[int, int, str, dict[str, Any], list[str]]] = []
        for family_code, rules in self.families.items():
            exclusions = [normalize_text(term) for term in rules.get("exclude", [])]
            matched_exclusions = [term for term in exclusions if term and term in normalized]
            if matched_exclusions:
                continue

            score = 0
            hits: list[str] = []
            for item in rules.get("include", []):
                term = normalize_text(item["term"])
                if term and term in normalized:
                    weight = int(item.get("weight", 1))
                    score += weight
                    hits.append(f"{item['term']} (+{weight})")

            if score >= self.minimum_score:
                # Menor priority vence em empate; mais hits funciona como critério adicional.
                priority = int(rules.get("priority", 999))
                candidates.append((score, -priority, family_code, rules, hits))

        if not candidates:
            return Classification(None, None, 0, "nenhuma regra atingiu o score mínimo", normalized)

        score, _, family_code, rules, hits = max(candidates, key=lambda x: (x[0], x[1], len(x[4])))
        return Classification(
            family_code=family_code,
            family_label=str(rules["label"]),
            score=score,
            reason="; ".join(hits),
            normalized_description=normalized,
        )

    def family_label(self, family_code: str) -> str | None:
        rules = self.families.get(family_code)
        return str(rules["label"]) if rules else None

    def apply_overrides(self, dataframe: pd.DataFrame, overrides_path: Path) -> pd.DataFrame:
        """Aplica inclusões/exclusões manuais por CATMAT após a regra textual."""
        if not overrides_path.exists() or overrides_path.stat().st_size == 0:
            return dataframe
        overrides = pd.read_csv(overrides_path, dtype={"codigo": "Int64"})
        if overrides.empty:
            return dataframe
        required = {"codigo", "acao", "familia_codigo"}
        missing = required - set(overrides.columns)
        if missing:
            raise ValueError(f"Colunas ausentes em material_overrides.csv: {sorted(missing)}")

        result = dataframe.copy()
        for _, override in overrides.iterrows():
            code = override.get("codigo")
            if pd.isna(code):
                continue
            mask = result["codigo"] == int(code)
            if not mask.any():
                continue
            action = str(override.get("acao", "")).strip().lower()
            note = str(override.get("observacao", "")).strip()
            if action == "excluir":
                result.loc[mask, ["familia_codigo", "familia_material"]] = None
                result.loc[mask, "classificacao_score"] = 0
                result.loc[mask, "classificacao_motivo"] = f"override manual: excluir — {note}"
                result.loc[mask, "material_relevante"] = False
            elif action == "incluir":
                family_code = str(override.get("familia_codigo", "")).strip()
                label = self.family_label(family_code)
                if not label:
                    raise ValueError(
                        f"Família inválida no override do CATMAT {int(code)}: {family_code}"
                    )
                result.loc[mask, "familia_codigo"] = family_code
                result.loc[mask, "familia_material"] = label
                result.loc[mask, "classificacao_score"] = 999
                result.loc[mask, "classificacao_motivo"] = f"override manual: incluir — {note}"
                result.loc[mask, "material_relevante"] = True
            else:
                raise ValueError(f"Ação inválida no override do CATMAT {int(code)}: {action}")
        return result

    def classify_dataframe(
        self,
        dataframe: pd.DataFrame,
        description_column: str = "descricao",
    ) -> pd.DataFrame:
        df = dataframe.copy()
        classifications = df[description_column].map(self.classify)
        df["descricao_normalizada"] = classifications.map(lambda x: x.normalized_description)
        df["familia_codigo"] = classifications.map(lambda x: x.family_code)
        df["familia_material"] = classifications.map(lambda x: x.family_label)
        df["classificacao_score"] = classifications.map(lambda x: x.score)
        df["classificacao_motivo"] = classifications.map(lambda x: x.reason)
        df["material_relevante"] = df["familia_codigo"].notna()
        return df
