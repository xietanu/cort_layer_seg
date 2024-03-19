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
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            key, value = line.split(" ")
            data[key] = DTYPE[key](value)
            
    return data