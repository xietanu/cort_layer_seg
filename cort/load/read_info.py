DTYPE = {
    "inplane_resolution_micron":float,
    "section_thickness_micron":float,
    "brain_id":str,
    "section_id":int,
    "patch_id":int,
    "brain_area":str,
}


def read_info(path: str) -> dict[str, object]:
    """Read info file for a cortical patch"""
    with open(path, "r", encoding="utf-8") as f:
        return process_info(f.readlines())


def process_info(lines: list[str]) -> dict[str, object]:
    """Process info file for a cortical patch"""
    data = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        key, value = line.split(" ")
        key = key.strip()
        value = value.strip()
        data[key] = DTYPE[key](value)
    return data
