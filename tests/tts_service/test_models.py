import pytest
from pydantic import ValidationError

from tts_service.models import TTSJobCreatePayload, TTSOutputOptions


def test_job_create_payload_uses_vizipvoice_defaults():
    payload = TTSJobCreatePayload.model_validate(
        {
            "text": "Xin chao anh.",
            "reference_text": "Day la transcript cua audio mau.",
        }
    )

    assert payload.text == "Xin chao anh."
    assert payload.reference_text == "Day la transcript cua audio mau."
    assert payload.generation.num_steps == 16
    assert payload.generation.guidance_scale == 1.0
    assert payload.generation.speed == 1.0
    assert payload.generation.t_shift == 0.5
    assert payload.generation.seed == 666
    assert payload.generation.max_duration_seconds == 100
    assert payload.generation.normalize_vietnamese is True
    assert payload.generation.split_sentences is True
    assert payload.generation.remove_long_silence is False
    assert payload.postprocess.crossfade_ms == 80
    assert payload.postprocess.silence_ms == 180
    assert payload.postprocess.fade_in_ms == 20
    assert payload.postprocess.fade_out_ms == 80
    assert payload.output.format == "mp3"


def test_payload_rejects_blank_text_and_reference_text():
    with pytest.raises(ValidationError):
        TTSJobCreatePayload.model_validate(
            {
                "text": "   ",
                "reference_text": "Day la transcript.",
            }
        )

    with pytest.raises(ValidationError):
        TTSJobCreatePayload.model_validate(
            {
                "text": "Noi dung can doc.",
                "reference_text": "",
            }
        )


def test_output_options_default_to_mp3_and_allow_wav_fallback():
    assert TTSOutputOptions.model_validate({}).format == "mp3"
    assert TTSOutputOptions.model_validate({"format": "mp3"}).format == "mp3"
    assert TTSOutputOptions.model_validate({"format": "wav"}).format == "wav"

    with pytest.raises(ValidationError):
        TTSOutputOptions.model_validate({"format": "flac"})


def test_generation_ranges_match_public_api_limits():
    with pytest.raises(ValidationError):
        TTSJobCreatePayload.model_validate(
            {
                "text": "Noi dung can doc.",
                "reference_text": "Transcript.",
                "generation": {"num_steps": 3},
            }
        )

    with pytest.raises(ValidationError):
        TTSJobCreatePayload.model_validate(
            {
                "text": "Noi dung can doc.",
                "reference_text": "Transcript.",
                "generation": {"speed": 2.0},
            }
        )
