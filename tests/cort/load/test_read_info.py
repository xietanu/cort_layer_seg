from cort.load.read_info import read_info, process_info

EXAMPLE_FILE_PATH = "tests/example_data/brain_region/section_a/0001/info.txt"

EXAMPLE_INFO = [
    "inplane_resolution_micron 1",
    "section_thickness_micron 20",
    "brain_id D20",
    "section_id 456",
    "patch_id 42",
    "brain_area not_real_area",
    "",
]


def test_process_info():
    data = process_info(EXAMPLE_INFO)
    assert data == {
        "inplane_resolution_micron": 1,
        "section_thickness_micron": 20,
        "brain_id": "D20",
        "section_id": 456,
        "patch_id": 42,
        "brain_area": "not_real_area",
    }, "process_info should return the correct dictionary"
    assert isinstance(
        data["inplane_resolution_micron"], float
    ), "inplane_resolution_micron should be a float"
    assert isinstance(
        data["section_thickness_micron"], float
    ), "section_thickness_micron should be a float"
    assert isinstance(data["brain_id"], str), "brain_id should be a string"
    assert isinstance(data["section_id"], int), "section_id should be an int"
    assert isinstance(data["patch_id"], int), "patch_id should be an int"
    assert isinstance(data["brain_area"], str), "brain_area should be a string"


def test_read_info():
    data = read_info(EXAMPLE_FILE_PATH)
    assert data == {
        "inplane_resolution_micron": 1,
        "section_thickness_micron": 20,
        "brain_id": "A24",
        "section_id": 123,
        "patch_id": 1,
        "brain_area": "fake_area",
    }, "read_info should return the correct dictionary"
    assert isinstance(
        data["inplane_resolution_micron"], float
    ), "inplane_resolution_micron should be a float"
    assert isinstance(
        data["section_thickness_micron"], float
    ), "section_thickness_micron should be a float"
    assert isinstance(data["brain_id"], str), "brain_id should be a string"
    assert isinstance(data["section_id"], int), "section_id should be an int"
    assert isinstance(data["patch_id"], int), "patch_id should be an int"
    assert isinstance(data["brain_area"], str), "brain_area should be a string"
