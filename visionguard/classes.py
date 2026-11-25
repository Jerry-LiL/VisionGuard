"""Defect labels shared by the dataset writer, the evaluator, and the API."""

DEFAULT_CLASS_NAMES = ("scratch", "crack", "spot")


def parse_class_names(
    value: str | None,
    default: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> tuple[str, ...]:
    if value is None or not value.strip():
        return default
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("class name list is empty")
    return names
