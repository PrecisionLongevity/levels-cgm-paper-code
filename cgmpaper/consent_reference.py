"""Read and validate an aggregate consent reference (consent-duration summary) without reading consent rows.

The reference is a JSON record with the fields version, participants, consent_summary (consent_median, consent_q25,
consent_q75, consent_overlapping_periods) and scope. It is an aggregate input derived upstream from the consent
records; this module checks its structure only and does not reconstruct consent periods.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

VERSION = 'upstream-consent-reference-v1'
_TOP_KEYS = {'version', 'participants', 'consent_summary', 'scope'}

_SUMMARY_KEYS = {'consent_median', 'consent_q25', 'consent_q75',
                 'consent_overlapping_periods'}


def _object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError('Consent reference contains a duplicate JSON key')
        record[key] = value
    return record


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f'Consent reference {name} must contain exactly the allowed fields')


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'Consent reference {name} must be a nonnegative integer')


def _duration(value, name):
    if type(value) not in (int, float) or value < 0:
        raise ValueError(f'Consent reference {name} must be finite and nonnegative')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'Consent reference {name} must be finite and nonnegative')


def validate_consent_summary(summary) -> dict:
    """Return the consent summary after checking its structural shape.

    Only schema fields are accepted. This does not reconstruct or revalidate
    the underlying consent periods.
    """
    _keys(summary, _SUMMARY_KEYS, 'summary')
    for key in ('consent_q25', 'consent_median', 'consent_q75'):
        _duration(summary[key], key)
    if not summary['consent_q25'] <= summary['consent_median'] <= summary['consent_q75']:
        raise ValueError('Consent reference quartiles must be ordered')
    _count(summary['consent_overlapping_periods'], 'overlapping-period count')
    return summary


def load_reference(path, *, expected_participants) -> dict:
    """Return the validated aggregate consent record for the expected number of participants.

    Only schema fields are accepted; extra fields (such as provenance labels) are ignored. Errors do not echo unexpected
    values.
    """
    _count(expected_participants, 'expected participant count')
    record = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_object)
    record = {k: record[k] for k in _TOP_KEYS if k in record}
    _keys(record, _TOP_KEYS, 'record')
    if record['version'] != VERSION:
        raise ValueError('Unsupported consent reference version')
    _count(record['participants'], 'participant count')
    if record['participants'] != expected_participants:
        raise ValueError('Consent reference participant count does not match the expected population')
    validate_consent_summary(record['consent_summary'])
    if not isinstance(record['scope'], str) or not record['scope'].strip():
        raise ValueError('Consent reference scope must be a nonempty string')
    return record
