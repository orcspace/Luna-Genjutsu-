import pytest

from genjutsu import prompts, providers, templates


def test_ten_templates_with_unique_ids():
    ids = [t["id"] for t in templates.TEMPLATES]
    assert len(ids) == 10 == len(set(ids))
    for t in templates.public():
        assert t["title"] and t["subtitle"] and len(t["colors"]) == 2
        assert "style" not in t  # prompt internals stay on the server


def test_required_input_and_wrapping():
    with pytest.raises(ValueError, match="Где"):
        templates.apply("location", {})
    out = templates.apply("location", {"tpl_input": "a beach at dawn"})
    assert out["mode"] == "motion" and out["scene"] == "a beach at dawn"
    out = templates.apply("outfit", {"tpl_input": "a red dress", "extra": "keep the earrings"})
    assert out["extra"].startswith("Dress the person in a red dress") and out["extra"].endswith("keep the earrings")
    out = templates.apply("object", {"tpl_input": "the mug"})
    assert out["mode"] == "object" and out["who"] == "the mug" and out["masking"] == "none"
    with pytest.raises(ValueError, match="шаблона"):
        templates.apply("nope", {})
    assert templates.apply(None, {})["template"] == "hero"


def test_user_masking_choice_wins_over_template_default():
    assert templates.apply("live", {"masking": "cloud"})["masking"] == "cloud"
    assert templates.apply("live", {})["masking"] == "none"


def test_style_templates_change_the_look_not_the_identity():
    opts = templates.apply("anime", {})
    text = prompts.build(opts["mode"], providers.Enhancor.tags, ["body", "face"], masked=True, style=opts["style"], photoreal=opts["photoreal"])
    assert "1990s Japanese cel anime" in text and "recognisable as the person from the references" in text
    assert "fully photorealistic" not in text
    plain = prompts.build("swap", providers.Enhancor.tags, ["face"], masked=True)
    assert "fully photorealistic" in plain and "Target style" not in plain


def test_object_prompt():
    text = prompts.build("object", providers.Enhancor.tags, ["object"], who="the phone in her hand")
    assert text.startswith("Replace only the phone in her hand in @Video1")
    assert "@Image1 shows the new object" in text and "Preserve everything else in @Video1" in text
    assert "face morphing" not in text
