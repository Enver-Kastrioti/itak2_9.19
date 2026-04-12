#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatchRecord:
    sequence_id: str
    name: str
    family: str
    type: str
    desc: tuple
    other_family: str
    rule_id: str = "NA"
    matched_iprs: tuple = ()
    matched_accessions: tuple = ()
    matched_libraries: tuple = ()
    matched_domain_count: int = 0
    evidence_summary: str = "NA"
    evidence_hits: tuple = ()
    pk_shiu_class: str = "NA"
    pk_ppc_class: str = "NA"
    pk_ppc_description: str = "NA"
    sequence: str = ""


def build_tftr_match_record(sequence_id, data):
    desc = tuple(data.get("desc", []) or [])
    return MatchRecord(
        sequence_id=sequence_id,
        rule_id=data.get("rule_id", "NA"),
        name=data.get("name", "NA"),
        family=data.get("family", "NA"),
        type=data.get("type", "NA"),
        desc=desc,
        other_family=data.get("other_family", "NA"),
        matched_iprs=tuple(data.get("matched_iprs", []) or []),
        matched_accessions=tuple(data.get("matched_accessions", []) or []),
        matched_libraries=tuple(data.get("matched_libraries", []) or []),
        matched_domain_count=int(data.get("matched_domain_count", 0) or 0),
        evidence_summary=data.get("evidence_summary", "NA") or "NA",
        evidence_hits=tuple(data.get("evidence_hits", []) or []),
    )


def build_pk_match_record(sequence_id, shiu_class, ppc_class, ppc_description, sequence=""):
    return MatchRecord(
        sequence_id=sequence_id,
        name=ppc_class,
        family=ppc_class,
        type="PK",
        desc=tuple([ppc_description] if ppc_description and ppc_description != "NA" else []),
        other_family=f"Shiu:{shiu_class}",
        pk_shiu_class=shiu_class or "NA",
        pk_ppc_class=ppc_class or "NA",
        pk_ppc_description=ppc_description or "NA",
        sequence=sequence or "",
    )


def records_to_legacy_json(records):
    return {
        seq_id: {
            "name": record.name,
            "family": record.family,
            "type": record.type,
            "desc": list(record.desc),
            "other_family": record.other_family,
            "rule_id": record.rule_id,
            "matched_iprs": list(record.matched_iprs),
            "matched_accessions": list(record.matched_accessions),
            "matched_libraries": list(record.matched_libraries),
            "matched_domain_count": record.matched_domain_count,
            "evidence_summary": record.evidence_summary,
            "evidence_hits": list(record.evidence_hits),
        }
        for seq_id, record in sorted(records.items())
    }


def records_to_classification_result(records):
    return records_to_legacy_json(records)


def _display_value(value, default="NA"):
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"na", "null", "none"}:
        return default
    return text


def write_tftr_outputs(records, output_dir, debug=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    match_tbl = output_dir / "match_tbl.txt"
    evidence_tsv = output_dir / "tftr_domain_evidence.tsv"
    with open(match_tbl, "w", encoding="utf-8") as handle:
        handle.write(
            "Sequence_ID\tName\tFamily\tType\tDescription\tOther_Family\tRule_ID\tMatched_IPR_IDs\tMatched_Accessions\tMatched_Libraries\tMatched_Domain_Count\tEvidence_Summary\n"
        )
        for seq_id, record in sorted(records.items()):
            desc_str = ";".join(record.desc) if record.desc else "NA"
            handle.write(
                f"{seq_id}\t{record.name}\t{record.family}\t{record.type}\t{desc_str}\t{record.other_family}\t"
                f"{record.rule_id}\t"
                f"{';'.join(record.matched_iprs) if record.matched_iprs else 'NA'}\t"
                f"{';'.join(record.matched_accessions) if record.matched_accessions else 'NA'}\t"
                f"{';'.join(record.matched_libraries) if record.matched_libraries else 'NA'}\t"
                f"{record.matched_domain_count}\t"
                f"{record.evidence_summary}\n"
            )

    with open(evidence_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "Sequence_ID",
            "Evidence_Source",
            "Domain_Key",
            "IPR",
            "Accession",
            "Library",
            "Description",
            "Start",
            "End",
            "Score",
            "Evalue",
        ])
        for seq_id, record in sorted(records.items()):
            for hit in record.evidence_hits:
                writer.writerow([
                    seq_id,
                    "TFTR",
                    _display_value(hit.get("domain_key", "")),
                    _display_value(hit.get("ipr", "")),
                    _display_value(hit.get("accession", "")),
                    _display_value(hit.get("library", "")),
                    _display_value(hit.get("description", "")),
                    _display_value(hit.get("start", "")),
                    _display_value(hit.get("end", "")),
                    _display_value(hit.get("score", "")),
                    _display_value(hit.get("evalue", "")),
                ])

    written = {"table": match_tbl, "evidence_tsv": evidence_tsv}
    if debug:
        match_json = output_dir / "match.json"
        with open(match_json, "w", encoding="utf-8") as handle:
            json.dump(records_to_legacy_json(records), handle, indent=2, ensure_ascii=False)
        written["json"] = match_json

    return written


def write_pk_outputs(records, output_dir, source_stem, debug=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classified_pk_fasta = output_dir / f"{source_stem}_pk_classified.fasta"
    pk_tsv = output_dir / "pk_classification.tsv"
    shiu_tsv = output_dir / "shiu_classification.txt"
    ppc_tsv = output_dir / "PPC_classification.txt"
    match_json = output_dir / "match.json"
    for stale_path in (output_dir / "pk_sequence.fasta", output_dir / "match_tbl.txt"):
        if stale_path.exists():
            stale_path.unlink()

    with open(classified_pk_fasta, "w", encoding="utf-8") as classified_handle, \
         open(pk_tsv, "w", encoding="utf-8") as pk_handle, \
         open(shiu_tsv, "w", encoding="utf-8") as shiu_handle, \
         open(ppc_tsv, "w", encoding="utf-8") as ppc_handle:
        pk_handle.write("Sequence_ID\tShiu_Class\tPPC_Class\tPPC_Description\n")
        for seq_id, record in sorted(records.items()):
            header = f">{seq_id} | {record.pk_ppc_class} | PK"
            classified_handle.write(f"{header}\n{record.sequence}\n")
            pk_handle.write(
                f"{seq_id}\t{record.pk_shiu_class}\t{record.pk_ppc_class}\t{record.pk_ppc_description}\n"
            )
            shiu_handle.write(f"{seq_id}\t{record.pk_shiu_class}\n")
            ppc_handle.write(f"{seq_id}\t{record.pk_ppc_class}\t{record.pk_ppc_description}\n")

    if debug:
        with open(match_json, "w", encoding="utf-8") as handle:
            json.dump(records_to_legacy_json(records), handle, indent=2, ensure_ascii=False)

    return {
        "classified_fasta": classified_pk_fasta,
        "combined_tsv": pk_tsv,
        "shiu_tsv": shiu_tsv,
        "ppc_tsv": ppc_tsv,
        "json": match_json if debug else None,
    }


def load_tftr_records_from_table(match_tbl_path):
    records = {}
    match_tbl_path = Path(match_tbl_path)
    if not match_tbl_path.exists():
        return records

    with open(match_tbl_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            seq_id, name, family, type_, desc, other_family = parts[:6]
            rule_id = parts[6] if len(parts) > 6 else "NA"
            matched_iprs = tuple([] if len(parts) <= 7 or parts[7] == "NA" else [item for item in parts[7].split(";") if item])
            matched_accessions = tuple([] if len(parts) <= 8 or parts[8] == "NA" else [item for item in parts[8].split(";") if item])
            matched_libraries = tuple([] if len(parts) <= 9 or parts[9] == "NA" else [item for item in parts[9].split(";") if item])
            matched_domain_count = int(parts[10]) if len(parts) > 10 and parts[10].isdigit() else 0
            evidence_summary = parts[11] if len(parts) > 11 else "NA"
            records[seq_id] = MatchRecord(
                sequence_id=seq_id,
                name=name,
                family=family,
                type=type_,
                desc=tuple([] if desc == "NA" else [item for item in desc.split(";") if item]),
                other_family=other_family,
                rule_id=rule_id,
                matched_iprs=matched_iprs,
                matched_accessions=matched_accessions,
                matched_libraries=matched_libraries,
                matched_domain_count=matched_domain_count,
                evidence_summary=evidence_summary,
            )
    return records


def load_pk_records_from_tsv(pk_tbl_path):
    records = {}
    pk_tbl_path = Path(pk_tbl_path)
    if not pk_tbl_path.exists():
        return records

    with open(pk_tbl_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            seq_id = row.get("Sequence_ID")
            if not seq_id:
                continue
            shiu_class = row.get("Shiu_Class", "NA") or "NA"
            ppc_class = row.get("PPC_Class", "NA") or "NA"
            ppc_description = row.get("PPC_Description", "NA") or "NA"
            records[seq_id] = build_pk_match_record(
                sequence_id=seq_id,
                shiu_class=shiu_class,
                ppc_class=ppc_class,
                ppc_description=ppc_description,
            )
    return records


def write_combined_summary(tf_records, pk_records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(set(tf_records.keys()) | set(pk_records.keys()))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "Sequence_ID",
            "TFTR_Name",
            "TFTR_Family",
            "TFTR_Type",
            "TFTR_Description",
            "TFTR_Other_Family",
            "PK_Shiu_Class",
            "PK_PPC_Class",
            "PK_PPC_Description",
        ])
        for seq_id in all_ids:
            tf_record = tf_records.get(seq_id)
            pk_record = pk_records.get(seq_id)
            writer.writerow([
                seq_id,
                tf_record.name if tf_record else "NA",
                tf_record.family if tf_record else "NA",
                tf_record.type if tf_record else "NA",
                ";".join(tf_record.desc) if tf_record and tf_record.desc else "NA",
                tf_record.other_family if tf_record else "NA",
                pk_record.pk_shiu_class if pk_record else "NA",
                pk_record.pk_ppc_class if pk_record else "NA",
                pk_record.pk_ppc_description if pk_record else "NA",
            ])
