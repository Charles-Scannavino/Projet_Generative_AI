from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

from lxml import etree
from tqdm import tqdm


# Namespace Dublin Core utilisé dans les fichiers Service-Public
DC_NS = "http://purl.org/dc/elements/1.1/"


# Blocs situés à la racine du XML que l'on ne souhaite pas
# intégrer directement dans le texte du RAG.
#
# Ils correspondent principalement :
# - à la navigation ;
# - aux références ;
# - à des métadonnées ;
# - aux liens vers d'autres fiches.
EXCLUDED_TOP_LEVEL_TAGS = {
    "SurTitre",
    "Audience",
    "Canal",
    "FilDAriane",
    "Theme",
    "SousThemePere",
    "DossierPere",
    "RechercheGuideePere",
    "VoirAussi",
    "QuiPeutMAider",
    "Reference",
    "PourEnSavoirPlus",
    "NoticeLiee",
    "ServiceEnLigne",
    "RessourceWeb",
}


# Types d'éléments qui contiennent du texte utile
# pour notre futur moteur RAG.
TEXT_BLOCK_TAGS = {
    "Titre",
    "TitreRiche",
    "TitreFlottant",
    "Paragraphe",
    "Item",
    "Valeur",
    "Definition",
    "Attention",
    "ANoter",
    "ASavoir",
    "Exemple",
}


def local_tag(tag: str) -> str:
    """
    Retourne le nom local d'une balise XML.

    Exemple :
    {http://purl.org/dc/elements/1.1/}title
    devient :
    title
    """

    if not isinstance(tag, str):
        return str(tag)

    return tag.split("}")[-1]


def normalize_text(text: str | None) -> str:
    """
    Nettoie les espaces dans un texte.

    On remplace notamment les espaces insécables
    et les suites d'espaces par un seul espace.
    """

    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def text_of_element(element: etree._Element) -> str:
    """
    Récupère tout le texte contenu dans un élément XML,
    y compris le texte de ses sous-balises.

    Exemple :
    <Paragraphe>
        logement
        <MiseEnEvidence>social</MiseEnEvidence>
    </Paragraphe>

    devient :
    logement social
    """

    return normalize_text(
        " ".join(element.itertext())
    )


def dc_text(
    root: etree._Element,
    field: str,
) -> str:
    """
    Récupère une métadonnée Dublin Core.

    Exemples :
    dc:title
    dc:subject
    dc:description
    dc:date
    """

    value = root.findtext(
        f"{{{DC_NS}}}{field}"
    )

    return normalize_text(value)


def extract_text_blocks(
    root: etree._Element,
) -> list[str]:
    """
    Extrait les blocs textuels utiles d'une fiche Service-Public.

    La déduplication ne s'applique qu'aux balises conteneurs
    susceptibles de répéter le texte de leurs enfants.
    """

    blocks: list[str] = []
    seen: set[str] = set()

    wrapper_tags = {
        "ASavoir",
        "Attention",
        "ANoter",
        "Exemple",
        "Definition",
    }

    for child in root:

        tag = local_tag(child.tag)

        # Ignore les métadonnées Dublin Core.
        if (
            isinstance(child.tag, str)
            and child.tag.startswith(f"{{{DC_NS}}}")
        ):
            continue

        # Ignore navigation, références et liens annexes.
        if tag in EXCLUDED_TOP_LEVEL_TAGS:
            continue

        for element in child.iter():

            element_tag = local_tag(element.tag)

            if element_tag not in TEXT_BLOCK_TAGS:
                continue

            # On supprime uniquement les conteneurs qui
            # possèdent déjà des sous-blocs textuels.
            if element_tag in wrapper_tags:

                has_selected_descendant = any(
                    descendant is not element
                    and local_tag(descendant.tag) in TEXT_BLOCK_TAGS
                    for descendant in element.iter()
                    if isinstance(descendant.tag, str)
                )

                if has_selected_descendant:
                    continue

            text = text_of_element(element)

            if not text:
                continue

            if text in seen:
                continue

            seen.add(text)
            blocks.append(text)

    return blocks


def parse_publication(
    xml_bytes: bytes,
) -> dict:
    """
    Transforme une fiche XML Service-Public
    en document Python structuré.
    """

    parser = etree.XMLParser(
        recover=True,
        remove_comments=True,
        huge_tree=True,
    )

    root = etree.fromstring(
        xml_bytes,
        parser=parser,
    )

    # Identifiant officiel de la fiche
    document_id = (
        root.attrib.get("ID")
        or dc_text(root, "identifier")
    )

    # Métadonnées Dublin Core
    title = dc_text(root, "title")
    subject = dc_text(root, "subject")
    description = dc_text(root, "description")
    date = dc_text(root, "date")
    document_type = dc_text(root, "type")
    coverage = dc_text(root, "coverage")

    # Métadonnées présentes sur la racine XML
    url = root.attrib.get("spUrl", "")

    important_modified_date = root.attrib.get(
        "dateDerniereModificationImportante",
        "",
    )

    # Extraction du contenu textuel
    blocks = extract_text_blocks(root)

    text = "\n\n".join(blocks)

    return {
        "document_id": document_id,
        "title": title,
        "subject": subject,
        "description": description,
        "date": date,
        "important_modified_date": important_modified_date,
        "url": url,
        "document_type": document_type,
        "coverage": coverage,
        "text": text,
        "n_chars": len(text),
        "n_words": len(text.split()),
        "n_blocks": len(blocks),
    }


def iter_fiche_members(
    zip_file: ZipFile,
) -> Iterable[str]:
    """
    Retourne uniquement les fichiers F*.xml.

    Les autres XML du ZIP ne correspondent pas
    aux fiches Service-Public que nous souhaitons utiliser.
    """

    for member in zip_file.namelist():

        filename = Path(member).name

        if (
            filename.startswith("F")
            and filename.endswith(".xml")
        ):
            yield member


def load_documents(
    zip_path: Path,
    subject_filter: str | None = None,
    ids: set[str] | None = None,
) -> list[dict]:
    """
    Lit les fiches du ZIP et applique éventuellement :

    - un filtre par sujet ;
    - un filtre par identifiants.
    """

    documents: list[dict] = []

    with ZipFile(zip_path) as z:

        members = list(
            iter_fiche_members(z)
        )

        for member in tqdm(
            members,
            desc="Parsing Service-Public",
        ):

            xml_bytes = z.read(member)

            try:

                document = parse_publication(
                    xml_bytes
                )

            except Exception as exc:

                print(
                    f"[WARN] Impossible de parser "
                    f"{member}: {exc}"
                )

                continue

            document_id = document[
                "document_id"
            ]

            # Filtre sur une liste précise d'identifiants
            if (
                ids
                and document_id not in ids
            ):
                continue

            # Filtre sur le sujet
            if subject_filter:

                subject = (
                    document["subject"]
                    .casefold()
                )

                if (
                    subject_filter.casefold()
                    not in subject
                ):
                    continue

            documents.append(document)

    return documents


def save_jsonl(
    documents: list[dict],
    output_path: Path,
) -> None:
    """
    Sauvegarde les documents au format JSONL.

    Une ligne = un document.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for document in documents:

            line = json.dumps(
                document,
                ensure_ascii=False,
            )

            f.write(
                line + "\n"
            )


def print_stats(
    documents: list[dict],
) -> None:
    """
    Affiche les premiers contrôles qualité du corpus.
    """

    print()
    print("=" * 70)
    print("STATISTIQUES")
    print("=" * 70)

    print(
        "Nombre de documents :",
        len(documents),
    )

    if not documents:
        return

    lengths = [
        doc["n_chars"]
        for doc in documents
    ]

    word_counts = [
        doc["n_words"]
        for doc in documents
    ]

    empty = [
        doc
        for doc in documents
        if not doc["text"].strip()
    ]

    print(
        "Documents vides :",
        len(empty),
    )

    print(
        "Longueur minimale (caractères) :",
        min(lengths),
    )

    print(
        "Longueur maximale (caractères) :",
        max(lengths),
    )

    print(
        "Longueur moyenne (caractères) :",
        round(
            sum(lengths) / len(lengths),
            1,
        ),
    )

    print(
        "Nombre minimal de mots :",
        min(word_counts),
    )

    print(
        "Nombre maximal de mots :",
        max(word_counts),
    )

    print(
        "Nombre moyen de mots :",
        round(
            sum(word_counts)
            / len(word_counts),
            1,
        ),
    )

    subjects = Counter(
        doc["subject"]
        for doc in documents
    )

    print()
    print("Sujets :")

    for subject, count in subjects.most_common():

        print(
            f"  {subject}: {count}"
        )


def main() -> None:
    """
    Point d'entrée du script.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Parse les fiches XML Service-Public "
            "contenues dans vosdroits-latest.zip"
        )
    )

    parser.add_argument(
        "--zip",
        default=(
            "data/raw/service_public/"
            "vosdroits-latest.zip"
        ),
        help=(
            "Chemin vers le ZIP Service-Public"
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "data/processed/"
            "service_public_documents.jsonl"
        ),
        help=(
            "Fichier JSONL de sortie"
        ),
    )

    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Filtre optionnel sur dc:subject"
        ),
    )

    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help=(
            "Liste optionnelle d'identifiants "
            "Service-Public, par exemple "
            "F10007 F18005"
        ),
    )

    args = parser.parse_args()

    ids = (
        set(args.ids)
        if args.ids
        else None
    )

    documents = load_documents(
        zip_path=Path(args.zip),
        subject_filter=args.subject,
        ids=ids,
    )

    save_jsonl(
        documents,
        Path(args.out),
    )

    print_stats(documents)

    print()
    print(
        "Fichier écrit :",
        args.out,
    )


if __name__ == "__main__":
    main()
