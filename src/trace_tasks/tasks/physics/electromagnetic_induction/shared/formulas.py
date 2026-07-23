"""Physical rules for electromagnetic induction panels."""


def induced_current_class(field_orientation: str, flux_change: str) -> str:
    """Return current class from Lenz-law field opposition."""

    if str(flux_change) == "none":
        return "no_current"
    induced_field = ""
    if str(flux_change) == "increasing" and str(field_orientation) == "into_page":
        induced_field = "out_of_page"
    if str(flux_change) == "increasing" and str(field_orientation) == "out_of_page":
        induced_field = "into_page"
    if str(flux_change) == "decreasing":
        induced_field = str(field_orientation)
    return "clockwise" if str(induced_field) == "into_page" else "counterclockwise"
