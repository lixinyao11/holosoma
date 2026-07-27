from pathlib import Path

from holosoma.config_values.terrain import terrain_load_obj
from holosoma.config_values.wbt.g1.command import motion_config_climb
from holosoma.utils.path import resolve_data_file_path


def test_registered_climb_motion_exists_in_package_data():
    assert Path(resolve_data_file_path(motion_config_climb.motion_file)).is_file()


def test_registered_obj_terrain_exists_in_package_data():
    obj_file_path = terrain_load_obj.terrain_term.obj_file_path
    assert obj_file_path is not None
    assert Path(resolve_data_file_path(obj_file_path)).is_file()
