from __future__ import annotations

import numpy as np

from sceneledger.models.decode import decode_slot_arrays
from sceneledger.types import Span


def test_decode_slot_arrays_enforces_presence_pointer_and_containment() -> None:
    outputs = {
        "track_presence_logits": np.array([[5.0, -5.0]]),
        "track_type_logits": np.array([[[0.0, 0.0, 5.0, 0.0, 0.0, 0.0], [5.0] * 6]]),
        "track_activity_logits": np.array(
            [[[5.0] * 6 + [-5.0] * 4, [-5.0] * 10]]
        ),
        "track_audibility_logits": np.array([[4.0, -4.0]]),
        "eventness_logits": np.array([[5.0, -5.0]]),
        "event_type_logits": np.array([[[0.0, 0.0, 5.0, 0.0], [5.0] * 4]]),
        "event_activity_logits": np.array(
            [[[-5.0, -5.0, 5.0, 5.0, 5.0, 5.0, 5.0, -5.0, -5.0, -5.0], [-5.0] * 10]]
        ),
        "track_pointer_logits": np.array([[[5.0, -5.0, -5.0], [-5.0, -5.0, 5.0]]]),
        "onset_logits": np.zeros((1, 2, 10)),
        "offset_logits": np.zeros((1, 2, 10)),
    }
    ledger = decode_slot_arrays(
        outputs,
        sample_id="decoded",
        duration_sec=1.0,
        event_texts={0: "soft music"},
    )
    assert len(ledger.tracks) == 1
    assert ledger.tracks[0].kind == "music"
    assert len(ledger.events) == 1
    assert ledger.events[0].track_id == "T1"
    assert ledger.events[0].spans == [Span(0.2, 0.6)]
