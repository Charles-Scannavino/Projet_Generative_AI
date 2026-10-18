"""Erreurs typées des outils déterministes.

Les messages sont rédigés par nous et ne recopient jamais la valeur brute
reçue : ils sont renvoyés tels quels au LLM par l'agent, et une valeur
injectée ne doit pas pouvoir y revenir sous forme d'instruction.
"""

from __future__ import annotations


class ToolError(ValueError):
    """Base des erreurs « attendues » d'un outil, présentables à l'utilisateur.

    Sous-classe de ValueError : les appelants qui traitent déjà les entrées
    invalides (`except ValueError`, agent à formulaire JSON) les attrapent aussi.
    """

    code = "tool_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


class ToolInputError(ToolError):
    code = "invalid_input"


class CommuneNotFound(ToolError):
    code = "commune_not_found"

    def __init__(self) -> None:
        super().__init__(
            "Commune introuvable dans le Code officiel géographique. "
            "Vérifier l'orthographe ou fournir le code Insee."
        )


class AmbiguousCommune(ToolError):
    code = "ambiguous_commune"

    def __init__(self, candidates: list[dict]) -> None:
        self.candidates = candidates
        listing = ", ".join(f"{c['libelle']} ({c['departement']})" for c in candidates)
        super().__init__(
            f"Plusieurs communes portent ce nom : {listing}. Préciser le département."
        )

    def to_dict(self) -> dict:
        return {**super().to_dict(), "candidates": self.candidates}


class DataNotCovered(ToolError):
    code = "data_not_covered"
