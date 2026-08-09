"""Deterministic serialization of a :class:`Ledger` to/from XML caption.

The JSON Schema (``schemas/track_event_ledger.schema.json``) is the canonical
on-disk representation; this module provides the human- and model-facing XML
caption. Two modes:

* ``mode="full"``  -- lossless: tracks + events + conditions + provenance +
  relations + evidence + attributes + span uncertainties. Used for storage and
  the strict round-trip test.
* ``mode="events"`` -- the caption the model is asked to emit: events only,
  with a minimal set of readable attributes. Track identities are inlined as
  ``identity`` so the caption is self-describing without a tracks header.

``serialize`` / ``deserialize`` are strict inverses in ``full`` mode.
``sceneledger.eval.parser`` provides the tolerant counterpart that consumes
raw model output.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable

from sceneledger.data.schema import (
    SCHEMA_VERSION,
    TIME_RESOLUTION_SEC,
    Conditions,
    Event,
    Evidence,
    Ledger,
    Provenance,
    Relation,
    Span,
    Track,
)

# Tag names for event types are fixed (these ARE the <speech>/<lys>/... tags).
_EVENT_TAGS: tuple[str, ...] = ("speech", "lys", "music", "sfx")
_ATTR_TRUE = "true"
_ATTR_FALSE = "false"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fmt_span(span: Span) -> str:
    return f"{span.start_sec:g}-{span.end_sec:g}"


def _fmt_span_list(spans: Iterable[Span]) -> str:
    return ",".join(_fmt_span(s) for s in spans)


def _fmt_uncertainty_list(spans: Iterable[Span]) -> str | None:
    """Encode per-span ``start|end`` uncertainties; None if all absent."""
    parts: list[str] = []
    has_any = False
    for s in spans:
        su = "" if s.start_uncertainty_sec is None else f"{s.start_uncertainty_sec:g}"
        eu = "" if s.end_uncertainty_sec is None else f"{s.end_uncertainty_sec:g}"
        if su or eu:
            has_any = True
        parts.append(f"{su}|{eu}")
    return ";".join(parts) if has_any else None


def _parse_span_list(text: str) -> list[list[float | None]]:
    """Parse ``"0.4-0.8,1.6-2.0"`` into mutable span rows.

    Each row is ``[start, end, start_unc, end_unc]``; uncertainties are filled
    later by :func:`_apply_uncertainties`.
    """
    spans: list[list[float | None]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"invalid span chunk {chunk!r} (expected start-end)")
        # negative start not allowed by schema; safe to split on first '-'
        start_s, end_s = chunk.split("-", 1)
        spans.append([float(start_s), float(end_s), None, None])
    return spans


def _apply_uncertainties(
    spans: list[list[float | None]],
    tu_text: str | None,
) -> None:
    """Fill uncertainty values from a ``"su|eu;su|eu"`` sidecar string."""
    if tu_text is None:
        return
    chunks = tu_text.split(";")
    if len(chunks) != len(spans):
        raise ValueError(
            f"uncertainty count {len(chunks)} != span count {len(spans)}"
        )
    for span_row, chunk in zip(spans, chunks, strict=False):
        su_s, eu_s = chunk.split("|", 1)
        span_row[2] = float(su_s) if su_s else None
        span_row[3] = float(eu_s) if eu_s else None


def _make_span_objects(
    spans: list[list[float | None]],
) -> list[Span]:
    return [
        Span(
            start_sec=row[0],  # type: ignore[arg-type]
            end_sec=row[1],  # type: ignore[arg-type]
            start_uncertainty_sec=row[2],  # type: ignore[arg-type]
            end_uncertainty_sec=row[3],  # type: ignore[arg-type]
        )
        for row in spans
    ]


def _attr_to_subelement(parent: ET.Element, k: str, v: str | float | bool | None) -> None:
    """Append ``<attr k=... v=... [type=...]/>`` with type info for non-strings."""
    if v is None:
        return
    a = ET.SubElement(parent, "attr")
    a.set("k", k)
    if isinstance(v, bool):
        a.set("v", _ATTR_TRUE if v else _ATTR_FALSE)
        a.set("type", "bool")
    elif isinstance(v, float):
        a.set("v", f"{v:g}")
        a.set("type", "float")
    else:
        a.set("v", str(v))


def _attrs_from_element(parent: ET.Element) -> dict[str, str | float | bool | None]:
    out: dict[str, str | float | bool | None] = {}
    for a in parent.findall("attr"):
        k = a.get("k")
        if k is None:
            continue
        v = a.get("v", "")
        t = a.get("type")
        if t == "float":
            out[k] = float(v)
        elif t == "bool":
            out[k] = v == _ATTR_TRUE
        else:
            out[k] = v
    return out


def _bool_attr(v: bool | None) -> str | None:
    if v is None:
        return None
    return _ATTR_TRUE if v else _ATTR_FALSE


def _opt(v: float | str | None) -> str | None:
    return None if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


# --------------------------------------------------------------------------- #
# serialize
# --------------------------------------------------------------------------- #
def _spans_to_attrs(spans: list[Span]) -> dict[str, str]:
    attrs: dict[str, str] = {"t": _fmt_span_list(spans)}
    tu = _fmt_uncertainty_list(spans)
    if tu is not None:
        attrs["tu"] = tu
    return attrs


def _evidence_to_elem(evidence: Evidence, tag: str = "evidence") -> ET.Element | None:
    el = ET.Element(tag)
    if evidence.method is not None:
        el.set("method", evidence.method)
    if evidence.audio_support is not None:
        el.set("audio_support", f"{evidence.audio_support:g}")
    if evidence.target_residual_margin is not None:
        el.set("target_residual_margin", _opt(evidence.target_residual_margin))
    if evidence.av_support is not None:
        el.set("av_support", f"{evidence.av_support:g}")
    if evidence.waveform_uri is not None:
        el.set("waveform_uri", evidence.waveform_uri)
    if evidence.mask_uri is not None:
        el.set("mask_uri", evidence.mask_uri)
    # only emit if non-empty
    return el if dict(el.attrib) else None


def _evidence_from_elem(el: ET.Element | None) -> Evidence | None:
    if el is None:
        return None
    kw: dict = {}
    if "method" in el.attrib:
        kw["method"] = el.get("method")
    if "audio_support" in el.attrib:
        kw["audio_support"] = float(el.get("audio_support"))  # type: ignore[arg-type]
    if "target_residual_margin" in el.attrib:
        kw["target_residual_margin"] = float(el.get("target_residual_margin"))  # type: ignore[arg-type]
    if "av_support" in el.attrib:
        kw["av_support"] = float(el.get("av_support"))  # type: ignore[arg-type]
    if "waveform_uri" in el.attrib:
        kw["waveform_uri"] = el.get("waveform_uri")
    if "mask_uri" in el.attrib:
        kw["mask_uri"] = el.get("mask_uri")
    return Evidence(**kw)


def _track_to_elem(track: Track) -> ET.Element:
    el = ET.Element("track")
    el.set("id", track.id)
    el.set("kind", track.kind)
    if track.identity is not None:
        el.set("identity", track.identity)
    el.set("confidence", f"{track.confidence:g}")
    if track.audibility is not None:
        el.set("audibility", f"{track.audibility:g}")
    if track.spans:
        el.attrib.update(_spans_to_attrs(track.spans))
    if track.evidence is not None:
        ev = _evidence_to_elem(track.evidence)
        if ev is not None:
            el.append(ev)
    for k, v in track.attributes.items():
        _attr_to_subelement(el, k, v)
    return el


def _event_to_elem(event: Event, identity: str | None) -> ET.Element:
    el = ET.Element(event.type)
    el.set("id", event.id)
    if event.track_id is not None:
        el.set("track", event.track_id)
    el.attrib.update(_spans_to_attrs(event.spans))
    el.set("confidence", f"{event.confidence:g}")
    if identity is not None:
        el.set("identity", identity)
    if event.verbatim is not None:
        el.set("verbatim", _bool_attr(event.verbatim))
    if event.language is not None:
        el.set("language", event.language)
    el.text = event.text
    if event.evidence is not None:
        ev = _evidence_to_elem(event.evidence)
        if ev is not None:
            el.append(ev)
    for r in event.relations:
        rel = ET.SubElement(el, "rel")
        rel.set("pred", r.predicate)
        rel.set("target", r.target_event_id)
    for k, v in event.attributes.items():
        _attr_to_subelement(el, k, v)
    return el


def _conditions_to_elem(c: Conditions) -> ET.Element | None:
    el = ET.Element("conditions")
    for field in ("domain", "snr_db", "t60_sec", "echo", "codec", "overlap_ratio"):
        v = getattr(c, field)
        if v is None:
            continue
        el.set(field, _bool_attr(v) if isinstance(v, bool) else _opt(v))  # type: ignore[arg-type]
    return el if dict(el.attrib) else None


def _provenance_to_elem(p: Provenance) -> ET.Element | None:
    el = ET.Element("provenance")
    if p.label_level is not None:
        el.set("label_level", p.label_level)
    if p.source_dataset is not None:
        el.set("source_dataset", p.source_dataset)
    if p.renderer_manifest_uri is not None:
        el.set("renderer_manifest_uri", p.renderer_manifest_uri)
    if p.license_status is not None:
        el.set("license_status", p.license_status)
    for k, v in p.teacher_versions.items():
        tv = ET.SubElement(el, "teacher")
        tv.set("k", k)
        tv.set("v", v)
    return el if (dict(el.attrib) or list(el)) else None


def serialize(ledger: Ledger, mode: str = "full") -> str:
    """Serialize a :class:`Ledger` to an XML caption string.

    ``mode="full"`` is lossless; ``mode="events"`` emits only ``<events>``
    with inlined track identities (drops conditions/provenance/track-evidence/
    relations/attributes for readability).
    """
    root = ET.Element("ledger")
    root.set("schema_version", SCHEMA_VERSION)
    root.set("sample_id", ledger.sample_id)
    root.set("duration", f"{ledger.duration_sec:g}")
    root.set("time_resolution", f"{TIME_RESOLUTION_SEC:g}")
    if ledger.language is not None:
        root.set("language", ledger.language)

    identity_by_track = {t.id: t.identity for t in ledger.tracks}

    if mode == "full":
        cond = _conditions_to_elem(ledger.conditions)
        if cond is not None:
            root.append(cond)
        if ledger.tracks:
            tracks_el = ET.SubElement(root, "tracks")
            for t in ledger.tracks:
                tracks_el.append(_track_to_elem(t))
        events_el = ET.SubElement(root, "events")
        for e in ledger.events:
            events_el.append(_event_to_elem(e, identity_by_track.get(e.track_id)))
        prov = _provenance_to_elem(ledger.provenance)
        if prov is not None:
            root.append(prov)
    elif mode == "events":
        events_el = ET.SubElement(root, "events")
        for e in ledger.events:
            events_el.append(_event_to_elem(e, identity_by_track.get(e.track_id)))
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'full' or 'events'")

    _indent(root)
    return ET.tostring(root, encoding="unicode")


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print without extra blank lines (ET.indent in 3.9+)."""
    try:
        ET.indent(elem, space="  ")
    except AttributeError:  # pragma: no cover - py<3.9 not supported
        pass


# --------------------------------------------------------------------------- #
# deserialize (strict)
# --------------------------------------------------------------------------- #
def _spans_from_elem(el: ET.Element) -> list[Span]:
    if "t" not in el.attrib:
        raise ValueError(f"element <{el.tag}> missing 't' attribute")
    spans = _parse_span_list(el.get("t"))  # type: ignore[arg-type]
    _apply_uncertainties(spans, el.get("tu"))
    return _make_span_objects(spans)


def _track_from_elem(el: ET.Element) -> Track:
    if el.tag != "track":
        raise ValueError(f"expected <track>, got <{el.tag}>")
    attrs = dict(el.attrib)
    spans = _spans_from_elem(el)
    evidence = _evidence_from_elem(el.find("evidence"))
    attributes = _attrs_from_element(el)
    return Track(
        id=attrs["id"],
        kind=attrs["kind"],  # type: ignore[arg-type]
        identity=attrs.get("identity"),
        spans=spans,
        audibility=float(attrs["audibility"]) if "audibility" in attrs else None,
        confidence=float(attrs["confidence"]),
        evidence=evidence,
        attributes=attributes,
    )


def _event_from_elem(el: ET.Element) -> Event:
    if el.tag not in _EVENT_TAGS:
        raise ValueError(f"expected one of {_EVENT_TAGS}, got <{el.tag}>")
    attrs = dict(el.attrib)
    spans = _spans_from_elem(el)
    evidence = _evidence_from_elem(el.find("evidence"))
    relations = [
        Relation(predicate=r.get("pred"), target_event_id=r.get("target"))  # type: ignore[arg-type]
        for r in el.findall("rel")
    ]
    attributes = _attrs_from_element(el)
    verbatim_raw = attrs.get("verbatim")
    verbatim: bool | None
    if verbatim_raw is None:
        verbatim = None
    else:
        verbatim = verbatim_raw == _ATTR_TRUE
    text = (el.text or "").strip()
    if not text:
        raise ValueError(f"event {attrs.get('id')} has empty text")
    return Event(
        id=attrs["id"],
        type=el.tag,  # type: ignore[arg-type]
        track_id=attrs.get("track"),
        spans=spans,
        text=text,
        verbatim=verbatim,
        language=attrs.get("language"),
        confidence=float(attrs["confidence"]),
        evidence=evidence,
        relations=relations,
        attributes=attributes,
    )


def _conditions_from_elem(el: ET.Element) -> Conditions:
    kw: dict = {}
    if "domain" in el.attrib:
        kw["domain"] = el.get("domain")
    if "snr_db" in el.attrib:
        kw["snr_db"] = float(el.get("snr_db"))  # type: ignore[arg-type]
    if "t60_sec" in el.attrib:
        kw["t60_sec"] = float(el.get("t60_sec"))  # type: ignore[arg-type]
    if "echo" in el.attrib:
        kw["echo"] = el.get("echo") == _ATTR_TRUE
    if "codec" in el.attrib:
        kw["codec"] = el.get("codec")
    if "overlap_ratio" in el.attrib:
        kw["overlap_ratio"] = float(el.get("overlap_ratio"))  # type: ignore[arg-type]
    return Conditions(**kw)


def _provenance_from_elem(el: ET.Element) -> Provenance:
    kw: dict = {}
    if "label_level" in el.attrib:
        kw["label_level"] = el.get("label_level")
    if "source_dataset" in el.attrib:
        kw["source_dataset"] = el.get("source_dataset")
    if "renderer_manifest_uri" in el.attrib:
        kw["renderer_manifest_uri"] = el.get("renderer_manifest_uri")
    if "license_status" in el.attrib:
        kw["license_status"] = el.get("license_status")
    teacher_versions: dict[str, str] = {}
    for tv in el.findall("teacher"):
        teacher_versions[tv.get("k")] = tv.get("v")  # type: ignore[assignment]
    kw["teacher_versions"] = teacher_versions
    return Provenance(**kw)


def deserialize(text: str) -> Ledger:
    """Strictly parse an XML caption produced by :func:`serialize`.

    Raises ``ValueError`` (or ``ET.ParseError``) on any structural problem.
    For tolerant parsing of raw model output use
    :func:`sceneledger.eval.parser.parse_model_output`.
    """
    root = ET.fromstring(text)
    if root.tag != "ledger":
        raise ValueError(f"root element must be <ledger>, got <{root.tag}>")

    root_attrs = dict(root.attrib)
    schema_version = root_attrs.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, got {schema_version}"
        )

    sample_id = root_attrs["sample_id"]
    duration = float(root_attrs["duration"])
    language = root_attrs.get("language")

    conditions = Conditions()
    prov = Provenance()
    tracks: list[Track] = []
    events: list[Event] = []

    cond_el = root.find("conditions")
    if cond_el is not None:
        conditions = _conditions_from_elem(cond_el)

    tracks_el = root.find("tracks")
    if tracks_el is not None:
        tracks = [_track_from_elem(t) for t in tracks_el.findall("track")]

    events_el = root.find("events")
    if events_el is not None:
        for e in events_el:
            events.append(_event_from_elem(e))

    prov_el = root.find("provenance")
    if prov_el is not None:
        prov = _provenance_from_elem(prov_el)

    return Ledger(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        sample_id=sample_id,
        duration_sec=duration,
        time_resolution_sec=TIME_RESOLUTION_SEC,  # type: ignore[arg-type]
        language=language,
        conditions=conditions,
        tracks=tracks,
        events=events,
        provenance=prov,
    )


def events_to_caption(events: Iterable[Event], sample_id: str, duration: float) -> str:
    """Convenience: build an events-mode caption from a bare event list."""
    ledger = Ledger(
        sample_id=sample_id,
        duration_sec=duration,
        events=list(events),
    )
    return serialize(ledger, mode="events")


__all__ = [
    "serialize",
    "deserialize",
    "events_to_caption",
]
